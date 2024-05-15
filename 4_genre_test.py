#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py 
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, expr, lit, split, explode

def compute_popularity(ratings, movies):
    # Calculate average ratings for each movie
    movie_ratings = ratings.groupBy("movieId").agg(
        avg("rating").alias("avg_rating"),
        count("rating").alias("num_ratings")
    )
    # Join with movies dataset to get genres
    movie_ratings_with_genres = movie_ratings.join(movies, on="movieId")
    
    # Explode genres to get individual genres
    movie_ratings_with_genres = movie_ratings_with_genres.withColumn("genre", explode(split(col("genres"), "\\|")))
    
    # Calculate average rating and number of ratings per genre
    genre_popularity = movie_ratings_with_genres.groupBy("genre").agg(
        avg("avg_rating").alias("genre_avg_rating"),
        avg("num_ratings").alias("genre_num_ratings")
    )
    
    # Order by genre average rating and number of ratings
    top_genres = genre_popularity.orderBy(col("genre_avg_rating").desc(), col("genre_num_ratings").desc())
    
    return top_genres

def get_movie_id(top_genres, n_recommendations=100):
    # Limit the DataFrame to the top N genres and collect their IDs into a list
    return [row['genre'] for row in top_genres.limit(n_recommendations).collect()]

def compute_map(top_genres, ratings, n_recommendations=100):
    top_genre_id = get_movie_id(top_genres, n_recommendations)
    top_genre_id_expr = f"array({','.join([str(x) for x in top_genre_id])})"
    
    user_actual_genres = ratings.join(movies, on="movieId").withColumn("genre", explode(split(col("genres"), "\\|"))).groupBy("userId").agg(
        expr("collect_list(genre) as actual_genres")
    )
    
    precision_per_user = user_actual_genres.select(
        expr(f"size(array_intersect(actual_genres, {top_genre_id_expr})) as hits"),
        expr("size(actual_genres) as total_relevant"),
        lit(n_recommendations).alias("total_recommendations")
    ).selectExpr("hits / total_relevant as precision_at_k")
    
    mean_average_precision = precision_per_user.selectExpr("avg(precision_at_k) as MAP").first()['MAP']
    
    return mean_average_precision

def process_data(spark, userID):
    base_path = f'hdfs:///user/{userID}/ml-latest-small'
    train_path = f'{base_path}/train_ratings.csv'
    val_path = f'{base_path}/val_ratings.csv'
    test_path = f'{base_path}/test_ratings.csv'
    movies_path = f'{base_path}/movies.csv'
    
    train_ratings = spark.read.csv(train_path, header=True, inferSchema=True)
    val_ratings = spark.read.csv(val_path, header=True, inferSchema=True)
    test_ratings = spark.read.csv(test_path, header=True, inferSchema=True)
    movies = spark.read.csv(movies_path, header=True, inferSchema=True)
    
    top_genres = compute_popularity(train_ratings, movies)
    
    train_map = compute_map(top_genres, train_ratings)
    val_map = compute_map(top_genres, val_ratings)
    test_map = compute_map(top_genres, test_ratings)
    
    print(f"Train MAP: {train_map}, Validation MAP: {val_map}, Test MAP: {test_map}")

def main(spark, userID):
    process_data(spark, userID)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('q4_popularity_model').getOrCreate()
    userID = os.getenv('USER')
    main(spark, userID)

