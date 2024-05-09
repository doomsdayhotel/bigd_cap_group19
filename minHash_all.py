#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    $ spark-submit --deploy-mode client minHash_all.py ./ml-latest-small/ratings.csv
    $ spark-submit --deploy-mode client minHash.py ./ml-latest-small/ratings.csv
'''
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import collect_list, udf, col
from pyspark.ml.feature import MinHashLSH

def to_sparse_vector(movie_ids, total_movies):
    '''
    Notes
    1. indices need to start from 0
    2. indices need to be in a sorted order
    '''

    indices = [id-1 for id in movie_ids]
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
    movies_df = spark.read.csv(f'hdfs:/user/{userID}/ml-latest-small/movies.csv', header=True, inferSchema=True)

    print(movies_df.head())
    movies_df.show(5)

    # Get all unique movieIds
    unique_movie_ids = ratings_df.select("movieId").distinct().rdd.flatMap(lambda x: x).collect()
    total_movies = movies_df.agg(max(col("movieId"))).collect()[0][0]

    
    # Group by userId and collect all movieIds into a list
    ratings_df_grouped = ratings_df.groupBy("userId").agg(collect_list("newIndex").alias("movieIds"))
    # Show the transformed DataFrame
    ratings_df_grouped.show()
    # ratings_df_grouped.write.csv('hdfs:/user/hl5679_nyu_edu/ml-latest-small/ratings_df_grouped.csv', header=True, mode="overwrite")
    

    # Create UDF with total_movies bound
    to_sparse_vector_udf = udf(lambda ids: to_sparse_vector(ids, total_movies), VectorUDT())

    # Preping the Sparse Vector

    ratings_df_final = ratings_df_grouped.withColumn("features", to_sparse_vector_udf("movieIds"))
    ratings_df_final.show()
    # ratings_df_final.write.csv('hdfs:/user/hl5679_nyu_edu/ml-latest-small/ratings_df_final.csv', header=True, mode="overwrite")

    ''' 2. Applying MinHash '''
    # mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=10)
    # model = mh.fit(ratings_df_final)


    # transformed_df = model.transform(ratings_df_final)
    # similar_pairs = model.approxSimilarityJoin(transformed_df, transformed_df, 0.6, distCol="JaccardDistance")

    # similar_pairs = similar_pairs.filter("datasetA.userId < datasetB.userId")  # Avoid duplicates and self-pairs
    # sorted_pairs = similar_pairs.orderBy("JaccardDistance", ascending=True)
    # top_100_pairs = sorted_pairs.limit(100)
    # # top_100_pairs.select("datasetA.userId", "datasetB.userId", "JaccardDistance").show(100)
    # # top_100_pairs.printSchema()

    # simplified_df = top_100_pairs.select(
    #     col("datasetA.userId").alias("userIdA"),
    #     col("datasetB.userId").alias("userIdB"),
    #     "JaccardDistance"
    # )
    # # Write the simplified DataFrame to CSV
    # simplified_df.write.csv('hdfs:/user/hl5679_nyu_edu/ml-latest/top_100_simplified_pairs.csv', header=True, mode="overwrite")

    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('minHash').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
