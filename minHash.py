#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    $ spark-submit --deploy-mode client minHash.py ./ml-latest-small/ratings.csv
'''
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import collect_list, udf, col
from pyspark.sql import Row
from pyspark.sql.types import IntegerType

def to_sparse_vector(movie_ids, total_movies, movie_id_to_index):
    '''
    Notes
    1. indices need to start from 0
    2. indices need to be in a sorted order
    '''

    indices = sorted([movie_id_to_index[id]for id in movie_ids if id in movie_id_to_index])
    values = [1.0] * len(indices)
    return Vectors.sparse(total_movies, indices, values)


def main(spark, userID):
    '''
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    # 1. Preprocessing Data 
    # Load the ratings.csv into DataFrame
    ratings_df = spark.read.csv(f'hdfs:/user/{userID}/ml-latest-small/ratings.csv', header=True, inferSchema=True)
    
    # ratings_df = spark.read.csv("/Users/lvxinyuan/me/1-Projects/NYU/1-Courses/24_Spring_Big Data/hw/capstone-project-cap-19/ml-latest-small/ratings.csv")

    # Get all unique movieIds
    unique_movie_ids = ratings_df.select("movieId").distinct().rdd.flatMap(lambda x: x).collect()
    total_movies = len(unique_movie_ids) #9724

    # Create a dictionary for quick index lookup
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}

    # Convert the dictionary into a DataFrame
    movie_id_index_df = spark.createDataFrame(movie_id_to_index.items(), ["movieId", "index"])

    # Join the original DataFrame with the index DataFrame
    ratings_with_index_df = ratings_df.join(movie_id_index_df, on="movieId", how="left")

    ratings_with_index_df.show()

    
    # Group by userId and collect all movieIds into a list
    ratings_df_grouped = ratings_df.groupBy("userId").agg(collect_list("movieId").alias("movieIds"))
    # Show the transformed DataFrame
    # ratings_df_grouped.show()

    

    # Create UDF with total_movies bound
    to_sparse_vector_udf = udf(lambda ids: to_sparse_vector(ids, total_movies), VectorUDT())

    # Preping the Sparse Vector

    ratings_df_final = ratings_df_grouped.withColumn("features", to_sparse_vector_udf("movieIds"))
    ratings_df_final.show()

    ## 2. Applying MinHash


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('minHash').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
