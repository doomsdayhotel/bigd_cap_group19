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
    # First, calculate average ratings and the number of ratings for each movie
    movie_ratings = ratings.groupBy("movieId").agg(
        avg("rating").alias("avg_rating"),
        count("rating").alias("num_ratings")
    )
    
    # Ensure the DataFrame is materialized so that 'num_ratings' exists
    movie_ratings.cache()

    # Calculate the global average rating and the 90th percentile of the number of ratings
    global_average = ratings.agg(avg("rating")).first()[0]
    minimum_ratings = movie_ratings.stat.approxQuantile("num_ratings", [0.90], 0.0)[0]

    # Add a composite score column that uses both 'avg_rating' and 'num_ratings'
    composite_score = movie_ratings.withColumn(
        "composite_score",
        (col("num_ratings") / (col("num_ratings") + minimum_ratings) * col("avg_rating")) +
        (minimum_ratings / (col("num_ratings") + minimum_ratings) * global_average)
    )

    # Return the top movies sorted by the composite score
    top_movies = composite_score.orderBy(col("composite_score").desc())
    return top_movies


    return top_movies
def get_movie_id(top_movies, n_recommendations=100):
    ## Limit the DataFrame to the top N movies and collect their IDs into a list
    return [row['movieid'] for row in top_movies.limit(n_recommendations).collect()]

def compute_map(top_movies, ratings, n_recommendations=100):
    top_movie_id = get_movie_id(top_movies, n_recommendations)
    top_movie_id_expr = f"array({','.join([str(x) for x in top_movie_id])})"
    user_actual_movies = ratings.groupBy("userid").agg(
        expr("collect_list(movieid) as actual_movies")
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


