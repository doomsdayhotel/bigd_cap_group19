#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py <file_path>
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, expr, lit

def compute_popularity(ratings):
    # Calculate average ratings for each movie
    movie_ratings = ratings.groupBy("movieId").agg(
        avg("rating").alias("avg_rating"),
        count("rating").alias("num_ratings")
    )
    # Order by average rating and number of ratings (if have the same average rating, order by number of ratings)
    top_movies = movie_ratings.orderBy(col("avg_rating").desc(), col("num_ratings").desc())
    return top_movies

def get_movie_id(top_movies, n_recommendations=100):
    ## Limit the DataFrame to the top N movies and collect their IDs into a list
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
    base_path = f'hdfs:///user/{userID}/ml-latest'
    train_path = f'{base_path}/train_ratings.csv'
    val_path = f'{base_path}/val_ratings.csv'
    test_path = f'{base_path}/test_ratings.csv'
    
    train_ratings = spark.read.csv(train_path, header=True, inferSchema=True)
    val_ratings = spark.read.csv(val_path, header=True, inferSchema=True)
    test_ratings = spark.read.csv(test_path, header=True, inferSchema=True)

    top_movies = compute_popularity(train_ratings)
    train_map = compute_map(top_movies, train_ratings)
    val_map = compute_map(top_movies, val_ratings)
    test_map = compute_map(top_movies, test_ratings)

    print(f"Train MAP: {train_map}, Validation MAP: {val_map}, Test MAP: {test_map}")

def main(spark, userID):
    process_data(spark, userID)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('q4_popularity_model').getOrCreate()

    userID = os.getenv('USER')

    main(spark, userID)


