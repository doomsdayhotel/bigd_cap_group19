#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py <file_path>
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, expr, lit, split, explode, array, array_union, collect_list
from pyspark.ml.recommendation import ALS

def train_als_model(ratings):
    # Define ALS model
    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )
    # Train the model
    als_model = als.fit(ratings)
    return als_model

def get_top_n_recommendations(model, n_recommendations=100):
    # Generate top N movie recommendations for each user
    user_recs = model.recommendForAllUsers(n_recommendations)
    return user_recs

def get_movie_id(top_movies, n_recommendations=100):
    # Extract the top N movie IDs from the recommendations DataFrame
    top_movie_ids = top_movies.select(explode("recommendations.movieId").alias("movieId")).distinct().limit(n_recommendations).collect()
    return [row['movieId'] for row in top_movie_ids]

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
    ).selectExpr("hits / total_relevant as precision_at_k")
    
    mean_average_precision = precision_per_user.selectExpr("avg(precision_at_k) as MAP").first()['MAP']
    
    return mean_average_precision

def aggregate_genome_scores(genome_scores, genome_tags):
    # Join genome scores with genome tags to get tag names
    genome_scores_with_tags = genome_scores.join(genome_tags, on="tagId")
    
    # Aggregate the relevance scores for each movie
    movie_features = genome_scores_with_tags.groupBy("movieId").agg(
        collect_list(array("tag", "relevance")).alias("tag_relevance")
    )
    return movie_features

def process_data(spark):
    base_path = f'hdfs:///user/{os.getenv("USER")}/ml-latest'
    train_path = f'{base_path}/train_ratings.parquet'
    val_path = f'{base_path}/val_ratings.parquet'
    test_path = f'{base_path}/test_ratings.parquet'
    movies_path = f'{base_path}/movies.parquet'
    genome_scores_path = f'{base_path}/genome-scores.parquet'
    genome_tags_path = f'{base_path}/genome-tags.parquet'
    
    train_ratings = spark.read.parquet(train_path, header=True, inferSchema=True)
    val_ratings = spark.read.parquet(val_path, header=True, inferSchema=True)
    test_ratings = spark.read.parquet(test_path, header=True, inferSchema=True)
    movies = spark.read.parquet(movies_path, header=True, inferSchema=True)
    genome_scores = spark.read.parquet(genome_scores_path, header=True, inferSchema=True)
    genome_tags = spark.read.parquet(genome_tags_path, header=True, inferSchema=True)
    
    # Aggregate genome scores to create additional movie features
    movie_features = aggregate_genome_scores(genome_scores, genome_tags)
    
    # Join ratings with movie features
    train_ratings = train_ratings.join(movie_features, on="movieId", how="left")
    val_ratings = val_ratings.join(movie_features, on="movieId", how="left")
    test_ratings = test_ratings.join(movie_features, on="movieId", how="left")
    
    # Train ALS model
    als_model = train_als_model(train_ratings)
    
    # Get top N recommendations
    top_recommendations = get_top_n_recommendations(als_model)
    
    # Compute MAP
    print("Computing MAP on Training data")
    train_map = compute_map(top_recommendations, train_ratings)
    print(f"Train MAP: {train_map}")
    
    print("Computing MAP on Validation data")
    val_map = compute_map(top_recommendations, val_ratings)
    print(f"Validation MAP: {val_map}")
    
    print("Computing MAP on Test data")
    test_map = compute_map(top_recommendations, test_ratings)
    print(f"Test MAP: {test_map}")

def main(spark):
    process_data(spark)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('als_recommender').getOrCreate()
    main(spark)
