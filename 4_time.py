#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py <file_path>
"""

import os
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, lit, expr

def compute_decay_weighted_popularity(ratings, decay_rate):
    current_timestamp = ratings.agg({"timestamp": "max"}).collect()[0][0]
    
    # Calculate the age of each rating
    ratings_with_age = ratings.withColumn("age", (current_timestamp - col("timestamp")))

    # Apply the exponential decay function
    decay_expr = expr(f"exp(-{decay_rate} * age)").alias("decay_weight")
    ratings_with_decay = ratings_with_age.withColumn("decay_weight", decay_expr)
    ratings_with_decay = ratings_with_decay.withColumn("decayed_rating", col("rating") * col("decay_weight"))
    
    # Compute the weighted average rating for each movie
    movie_ratings = ratings_with_decay.groupBy("movieId").agg(
        avg("decayed_rating").alias("avg_decayed_rating"),
        count("rating").alias("num_ratings")
    )

    top_movies = movie_ratings.orderBy(col("avg_decayed_rating").desc())
    return top_movies

def get_movie_id(top_movies, n_recommendations=100):
    return [row['movieId'] for row in top_movies.limit(n_recommendations).collect()]

def compute_map(top_movies, ratings, n_recommendations=100):
    top_movie_id = get_movie_id(top_movies, n_recommendations)
    top_movie_id_expr = f"array({','.join([str(x) for x in top_movie_id])})"
    user_actual_movies = ratings.groupBy("userId").agg(
        expr("collect_list(movieId) as actual_movies")
    )
    precision_per_user = user_actual_movies.select(
        expr(f"size(array_intersect(actual_movies, {top_movie_id_expr})) as hits"),
        expr("size(actual_movies) as total_relevant"),
        lit(n_recommendations).alias("total_recommendations")
    ).selectExpr("hits / total_recommendations as precision_at_k")
    mean_average_precision = precision_per_user.selectExpr("avg(precision_at_k) as MAP").first()['MAP']
    return mean_average_precision

def process_data(spark, userID):
    base_path = f'hdfs://{userID}/ml-latest-small'
    train_path = f'{base_path}/train_ratings.csv'
    val_path = f'{base_path}/val_ratings.csv'
    test_path = f'{base_path}/test_ratings.csv'

    train_ratings = spark.read.csv(train_path, header=True, inferSchema=True)
    val_ratings = spark.read.csv(val_path, header=True, inferSchema=True)
    test_ratings = spark.read.csv(test_path, header=True, inferSchema=True)

    # Define decay rate
    decay_rate = 0.001  # Adjust this parameter based on your dataset

    top_movies = compute_decay_weighted_popularity(train_ratings, decay_rate)
    train_map = compute_map(top_movies, train_ratings)
    val_map = compute_map(top_movies, val_ratings)
    test_map = compute_map(top_movies, test_ratings)
    
    print(f"Train MAP: {train_map}, Validation MAP: {val_map}, Test MAP: {test_map}")

def main(spark, userID):
    process_data(spark, userID)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('q4_decay_weighted_popularity_model').getOrCreate()
    userID = os.getenv('USER')
    main(spark, userID)

