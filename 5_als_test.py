#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py <file_path>
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, expr, lit
from pyspark.ml.recommendation import ALS

def train_als_model(ratings, rank=10, maxIter=10, regParam=0.1):
    als = ALS(rank=rank, maxIter=maxIter, regParam=regParam, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(ratings)
    return model

def get_recommendations(model, user_id, n_recommendations=100):
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    user_recommendations = model.recommendForUserSubset(user_df, n_recommendations)
    recommendations = user_recommendations.selectExpr("explode(recommendations) as rec").selectExpr("rec.movieId as movieId")
    return recommendations.collect()

def compute_map(top_movies, ratings, n_recommendations=100):
    top_movie_id = [row['movieId'] for row in top_movies.limit(n_recommendations).collect()]
    top_movie_id_expr = f"array({','.join([str(x) for x in top_movie_id])})"
    user_actual_movies = ratings.groupBy("userId").agg(expr("collect_list(movieId) as actual_movies"))

    precision_per_user = user_actual_movies.select(
        expr(f"size(array_intersect(actual_movies, {top_movie_id_expr})) as hits"),
        expr("size(actual_movies) as total_relevant"),
        lit(n_recommendations).alias("total_recommendations")
    ).selectExpr("hits / total_recommendations as precision_at_k")

    mean_average_precision = precision_per_user.selectExpr("avg(precision_at_k) as MAP").first()["MAP"]
    return mean_average_precision

def process_data(spark, userID):
    base_path = f'hdfs:///user/{userID}/ml-latest-small'
    train_path = f'{base_path}/train_ratings.csv'
    val_path = f'{base_path}/val_ratings.csv'
    test_path = f'{base_path}/test_ratings.csv'

    train_ratings = spark.read.csv(train_path, header=True, inferSchema=True)
    val_ratings = spark.read.csv(val_path, header=True, inferSchema=True)
    test_ratings = spark.read.csv(test_path, header=True, inferSchema=True)

    als_model = train_als_model(train_ratings)

    train_map = compute_map(get_recommendations(als_model, userID, 100), train_ratings, 100)
    val_map = compute_map(get_recommendations(als_model, userID, 100), val_ratings, 100)
    test_map = compute_map(get_recommendations(als_model, userID, 100), test_ratings, 100)

    print(f"Train MAP: {train_map}, Validation MAP: {val_map}, Test MAP: {test_map}")

def main(spark, userID):
    process_data(spark, userID)

if __name__ == "__main__":
    spark = SparkSession.builder.appName("q4_collaborative_filtering_model").getOrCreate()
    userID = os.getenv('USER')
    main(spark, userID)
