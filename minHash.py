#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py
    $ spark-submit minHash.py ./ml-latest-small/ratings.csv
    $ spark-submit --deploy-mode client minHash.py ./ml-latest-small/ratings.csv
'''
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import collect_list
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT

def to_sparse_vector(movie_ids, total_movies):
    '''
    Notes
    1. indices need to start from 0
    2. indices need to be in a sorted order
    '''
    indices = [id for id in movie_ids] #sorted([id for id in movie_ids])
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
    # ratings_df = spark.read.csv(f'hdfs:/user/{userID}/ratings.csv')
    
    ratings_df = spark.read.csv("/Users/lvxinyuan/me/1-Projects/NYU/1-Courses/24_Spring_Big Data/hw/capstone-project-cap-19/ml-latest-small/ratings.csv", header=True, inferSchema=True)

    # Group by userId and collect all movieIds into a list
    ratings_df_grouped = ratings_df.groupBy("userId").agg(collect_list("movieId").alias("movieIds"))
    # Show the transformed DataFrame
    ratings_df_grouped.show()

    # Get all unique movieIds
    total_movies = ratings_df.select("movieId").distinct().count() #9724

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
