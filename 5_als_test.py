#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py <file_path>
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, size, collect_list, array_intersect
from pyspark.ml.evaluation import RegressionEvaluator
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

def evaluate_model_rmse(model, ratings):
    # Make predictions
    predictions = model.transform(ratings)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    return rmse

def get_top_n_recommendations(model, n_recommendations=100):
    # Generate top N movie recommendations for each user
    user_recs = model.recommendForAllUsers(n_recommendations)
    return user_recs

def compute_map(top_recommendations, ratings, n_recommendations=100):
    top_movie_id_expr = f"array({','.join([f'topRecommendations[{i}].movieId' for i in range(n_recommendations)])})"
    
    user_actual_movies = ratings.groupBy("userId").agg(
        expr("collect_list(movieId) as actual_movies")
    )

    precision_per_user = user_actual_movies.join(top_recommendations, "userId").select(
        col("userId"),
        expr(f"array_intersect(actual_movies, {top_movie_id_expr}) as hits"),
        size("actual_movies").alias("total_relevant"),
        expr(f"size(array_intersect(actual_movies, {top_movie_id_expr})) as hits"),
        size(top_movie_id_expr).alias("total_recommended"),
    ).selectExpr(
        "userId",
        "total_recommended",
        "total_relevant",
        "hits",
        "size(hits)/total_recommended as precision_at_k"
    )

    mean_average_precision = precision_per_user.selectExpr(
        "avg(precision_at_k) as MAP"
    ).first()["MAP"]
    
    return mean_average_precision

def process_data(spark):
    base_path = 'hdfs:///user/qy561_nyu_edu/ml-latest-small'
    train_path = f'{base_path}/train_ratings.csv'
    val_path = f'{base_path}/val_ratings.csv'
    test_path = f'{base_path}/test_ratings.csv'

    train_ratings = spark.read.csv(train_path, header=True, inferSchema=True)
    val_ratings = spark.read.csv(val_path, header=True, inferSchema=True)
    test_ratings = spark.read.csv(test_path, header=True, inferSchema=True)

    # Train ALS model
    als_model = train_als_model(train_ratings)

    # Evaluate the model
    print("Evaluating on Training data")
    train_rmse = evaluate_model_rmse(als_model, train_ratings)
    print(f"Train RMSE: {train_rmse}")
    print("Evaluating on Validation data")
    val_rmse = evaluate_model_rmse(als_model, val_ratings)
    print(f"Validation RMSE: {val_rmse}")
    print("Evaluating on Test data")
    test_rmse = evaluate_model_rmse(als_model, test_ratings)
    print(f"Test RMSE: {test_rmse}")

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

    return top_recommendations

def main(spark):
    top_recommendations = process_data(spark)
    print("Top Recommendations:", top_recommendations)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('als_recommender').getOrCreate()
    main(spark)
