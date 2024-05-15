#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py 
"""
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

def process_data(spark, userID):
    base_path = f'hdfs:///user/{userID}/ml-latest-small'
    train_path = f'{base_path}/train_ratings.csv'
    movies_path = f'{base_path}/movies.csv'
    
    train_ratings = spark.read.csv(train_path, header=True, inferSchema=True)
    movies = spark.read.csv(movies_path, header=True, inferSchema=True)
    
    top_genres = compute_popularity(train_ratings, movies)
    
    top_genres.show()

def main(spark, userID):
    process_data(spark, userID)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('q4_popularity_model').getOrCreate()
    userID = os.getenv('USER')
    main(spark, userID)

