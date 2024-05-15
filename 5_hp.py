#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client rec_sys.py <file_path>
"""

import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, expr, size, collect_list, explode, array_intersect, lit

def train_als_model_with_tuning(ratings):
    # Define ALS model
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )

    # Define parameter grid
    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [10, 20, 30]) \
        .addGrid(als.regParam, [0.01, 0.1, 0.2]) \
        .build()

    # Define evaluator
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    # Define cross-validator
    cross_validator = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5,
        parallelism=2
    )

    # Train the model with cross-validation
    cv_model = cross_validator.fit(ratings)

    return cv_model.bestModel

def get_top_n_recommendations(model, n_recommendations=100):
    user_recs = model.recommendForAllUsers(n_recommendations)
    return user_recs

def compute_map(top_movies, ratings, n_recommendations=100):
    movie_id_expr = get_movie_id(top_movies, n_recommendations)
    user_actual_movies = ratings.groupBy("userId").agg(
        expr("collect_list(movieId) as actual_movies")
    )
    
    precision_per_user = user_actual_movies.select(
        expr(f"""size(array_intersect(actual_movies, {movie_id_expr})) as hits"""),
        size(col("actual_movies")).alias("total_relevant"),
        lit(n_recommendations).alias("total_recommendations")
    ).selectExpr(
        "hits / total_recommendations as precision_at_k"
    )
    
    mean_average_precision = precision_per_user.selectExpr(
        "avg(precision_at_k) as MAP"
    ).first()["MAP"]
    
    return mean_average_precision

def get_movie_id(top_movies, n_recommendations=100):
    top_movie_ids = top_movies.select(explode("recommendations.movieId").alias("movieId")).distinct().limit(n_recommendations).collect()
    return f"array({','.join([str(row['movieId']) for row in top_movie_ids])})"

def process_data(spark, userID):
    base_path = f'hdfs:///user/{userID}/ml-latest-small'
    train_path = f'{base_path}/train_ratings.csv'
    val_path = f'{base_path}/val_ratings.csv'
    test_path = f'{base_path}/test_ratings.csv'

    train_ratings = spark.read.csv(train_path, header=True, inferSchema=True)
    val_ratings = spark.read.csv(val_path, header=True, inferSchema=True)
    test_ratings = spark.read.csv(test_path, header=True, inferSchema=True)

    als_model = train_als_model_with_tuning(train_ratings)
    top_recommendations = get_top_n_recommendations(als_model)

    train_map = compute_map(top_recommendations, train_ratings)
    print(f"Train MAP: {train_map}")
    val_map = compute_map(top_recommendations, val_ratings)
    print(f"Validation MAP: {val_map}")
    test_map = compute_map(top_recommendations, test_ratings)
    print(f"Test MAP: {test_map}")

    return top_recommendations

def main(spark, userID):
    process_data(spark, userID)

if __name__ == "__main__":
    spark = SparkSession.builder.appName("als_recommender").getOrCreate()
    userID = os.getenv('USER')
    main(spark, userID)
