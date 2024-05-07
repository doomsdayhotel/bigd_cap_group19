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
from pyspark.sql.functions import collect_list, udf
from pyspark.ml.feature import MinHashLSH

def to_sparse_vector(movie_ids, total_movies):
    '''
    Notes
    1. indices need to start from 0
    2. indices need to be in a sorted order
    '''

    indices = sorted(id for id in movie_ids)
    values = [1.0] * len(indices)
    return Vectors.sparse(total_movies, indices, values)


def main(spark, userID):
    '''
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    '''1. Preprocessing Data '''
    # Load the ratings.csv into DataFrame
    ratings_df = spark.read.csv(f'hdfs:/user/{userID}/ml-latest-small/ratings.csv', header=True, inferSchema=True)
    
    # ratings_df = spark.read.csv("/Users/lvxinyuan/me/1-Projects/NYU/1-Courses/24_Spring_Big Data/hw/capstone-project-cap-19/ml-latest-small/ratings.csv")

    # Get all unique movieIds
    unique_movie_ids = ratings_df.select("movieId").distinct().rdd.flatMap(lambda x: x).collect()
    total_movies = len(unique_movie_ids) #9724

    # Create a dictionary for quick index lookup
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}

    # Convert the dictionary into a DataFrame
    movie_id_index_df = spark.createDataFrame(movie_id_to_index.items(), ["movieId", "newIndex"])

    # Join the original DataFrame with the index DataFrame
    ratings_with_index_df = ratings_df.join(movie_id_index_df, on="movieId", how="left")

    # ratings_with_index_df.show()
    ratings_with_index_df.write.csv('hdfs:/user/{userID}/ml-latest-small', header=True, mode="overwrite")

    
    # Group by userId and collect all movieIds into a list
    ratings_df_grouped = ratings_with_index_df.groupBy("userId").agg(collect_list("newIndex").alias("movieIds"))
    # Show the transformed DataFrame
    # ratings_df_grouped.show()
    ratings_df_grouped.write.csv('hdfs:/user/{userID}/ml-latest-small', header=True, mode="overwrite")
    

    

    # Create UDF with total_movies bound
    to_sparse_vector_udf = udf(lambda ids: to_sparse_vector(ids, total_movies), VectorUDT())

    # Preping the Sparse Vector

    ratings_df_final = ratings_df_grouped.withColumn("features", to_sparse_vector_udf("movieIds"))
    # ratings_df_final.show()
    ratings_df_final.write.csv('hdfs:/user/{userID}/ml-latest-small', header=True, mode="overwrite")

    ''' 2. Applying MinHash '''
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=10)
    model = mh.fit(ratings_df_final)


    transformed_df = model.transform(ratings_df_final)
    similar_pairs = model.approxSimilarityJoin(transformed_df, transformed_df, 0.6, distCol="JaccardDistance")

    similar_pairs = similar_pairs.filter("datasetA.userId < datasetB.userId")  # Avoid duplicates and self-pairs
    sorted_pairs = similar_pairs.orderBy("JaccardDistance", ascending=False)
    top_100_pairs = sorted_pairs.limit(100)
    top_100_pairs.select("datasetA.userId", "datasetB.userId", "JaccardDistance").show()

    top_100_pairs.write.csv('hdfs:/user/{userID}/ml-latest-small', header=True, mode="overwrite")
    







# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('minHash').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
