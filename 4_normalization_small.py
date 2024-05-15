#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py <file_path>
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, lit, expr

def compute_mean_centered_popularity(ratings):
    # Step 1: Compute User Average Ratings
    user_avg_ratings = ratings.groupBy("userId").agg(avg("rating").alias("user_avg_rating"))
    
    # Step 2: Normalize Ratings by Subtracting User Average
    ratings = ratings.join(user_avg_ratings, on="userId")
    ratings = ratings.withColumn("normalized_rating", col("rating") - col("user_avg_rating"))
    
    # Step 3: Use Normalized Ratings in the Popularity Calculation
    movie_ratings = ratings.groupBy("movieId").agg(
        avg("normalized_rating").alias("avg_normalized_rating"),
        count("rating").alias("num_ratings")
    )
    
    top_movies = movie_ratings.orderBy(col("avg_normalized_rating").desc())
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
    base_path = f'hdfs:///user/{userID}/ml-latest-small'
    train_path = f'{base_path}/train_ratings.csv'
    val_path = f'{base_path}/val_ratings.csv'
    test_path = f'{base_path}/test_ratings.csv'

    train_ratings = spark.read.csv(train_path, header=True, inferSchema=True)
    val_ratings = spark.read.csv(val_path, header=True, inferSchema=True)
    test_ratings = spark.read.csv(test_path, header=True, inferSchema=True)

    top_movies = compute_mean_centered_popularity(train_ratings)
    train_map = compute_map(top_movies, train_ratings)
    val_map = compute_map(top_movies, val_ratings)
    test_map = compute_map(top_movies, test_ratings)
    
    print(f"Train MAP: {train_map}, Validation MAP: {val_map}, Test MAP: {test_map}")

def main(spark, userID):
    process_data(spark, userID)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('q4_mean_centered_popularity_model').getOrCreate()
    userID = os.getenv('USER')
    main(spark, userID)






base_path = f'hdfs:///user/{userID}/ml-latest-small'