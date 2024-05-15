#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py <file_path>
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, explode, split, lit, array, expr

def compute_clustered_popularity(ratings, movies):
    # Split the genres column into multiple rows
    movies_with_split_genres = movies.withColumn("genre", explode(split(col("genres"), "\|")))
    
    # Join Ratings with Movies Metadata
    ratings_with_movies = ratings.join(movies_with_split_genres, on="movieId")
    
    # Compute the average rating and count for each movie within each genre
    movie_cluster_ratings = ratings_with_movies.groupBy("movieId", "genre").agg(
        avg("rating").alias("avg_rating"),
        count("rating").alias("num_ratings")
    )
    
    # Compute the average rating across all clusters (genres) for each movie
    movie_ratings = movie_cluster_ratings.groupBy("movieId").agg(
        avg("avg_rating").alias("avg_cluster_rating"),
        count("genre").alias("num_genres")
    )
    
    # Sort movies by average cluster rating
    top_movies = movie_ratings.orderBy(col("avg_cluster_rating").desc())
    return top_movies

def get_movie_id(top_movies, n_recommendations=100):
    return [row['movieId'] for row in top_movies.limit(n_recommendations).collect()]

def compute_map(top_movies, ratings, n_recommendations=100):
    top_movie_id = get_movie_id(top_movies, n_recommendations)
    top_movie_id_expr = array(*[lit(x) for x in top_movie_id])  # Correct the array construction
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
    # Local paths for testing purposes
    base_path = f'hdfs:///user/{userID}/ml-latest-small'
    train_path = f'{base_path}/train_ratings.csv'
    val_path = f'{base_path}/val_ratings.csv'
    test_path = f'{base_path}/test_ratings.csv'
    movies_path = f'{base_path}/movies.csv'

    train_ratings = spark.read.csv(train_path, header=True, inferSchema=True)
    val_ratings = spark.read.csv(val_path, header=True, inferSchema=True)
    test_ratings = spark.read.csv(test_path, header=True, inferSchema=True)
    movies = spark.read.csv(movies_path, header=True, inferSchema=True)

    top_movies = compute_clustered_popularity(train_ratings, movies)
    train_map = compute_map(top_movies, train_ratings)
    val_map = compute_map(top_movies, val_ratings)
    test_map = compute_map(top_movies, test_ratings)
    
    print(f"Train MAP: {train_map}, Validation MAP: {val_map}, Test MAP: {test_map}")

def main(spark, userID):
    process_data(spark, userID)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('q4_clustered_popularity_model').getOrCreate()
    userID = os.getenv('USER')
    main(spark, userID)







