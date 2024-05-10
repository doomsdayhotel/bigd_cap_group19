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
from pyspark.sql.functions import collect_list, udf, col, max
from pyspark.ml.feature import CountVectorizer, MinHashLSH


def main(spark, userID):
    '''
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    '''1. Preprocessing Data '''
    # Load the ratings.csv into DataFrame
    ratings_df = spark.read.csv(f'hdfs:/user/{userID}/ml-latest/ratings.csv', schema='userId INT, movieId STRING, rating FLOAT, timestamp BIGINT')

    ratings_df.cache() #Cache for optimizing
    
    # Group by userId and collect all movieIds into a list
    ratings_df_grouped = ratings_df.groupBy("userId").agg(collect_list("movieId").alias("movieIds")).cache()
    ratings_df_grouped = ratings_df_grouped.repartition("userId")
    ratings_df_grouped.cache() #Cache for optimizing
    # ratings_df_grouped.show()
    '''
    +------+--------------------+
    |userId|            movieIds|
    +------+--------------------+
    |   148|[356, 1197, 4308,...|
    |   463|[110, 296, 356, 5...|
    |   471|[1, 296, 356, 527...|
    +------+--------------------+
    '''
    
    # Vectorize moviIds
    cv = CountVectorizer(inputCol = 'movieIds', outputCol = 'features')
    model = cv.fit(ratings_df_grouped)
    ratings_df_final = model.transform(ratings_df_grouped)
    # ratings_df_final.show()
    '''
    +------+--------------------+--------------------+
    |userId|            movieIds|            features|
    +------+--------------------+--------------------+
    |   148|[356, 1197, 4308,...|(9725,[0,19,25,26...|
    |   463|[110, 296, 356, 5...|(9725,[0,2,7,9,16...|
    |   471|[1, 296, 356, 527...|(9725,[0,2,4,9,10...|
    +------+--------------------+--------------------+
    '''

    ''' 2. Applying MinHash '''
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=1)
    model = mh.fit(ratings_df_final)

    print("Transformed Data\n")
    transformed_df = model.transform(ratings_df_final)
    similar_pairs = model.approxSimilarityJoin(transformed_df, transformed_df, 0.6, distCol="JaccardDistance")
    similar_pairs.cache()
    similar_pairs.write.partitionBy("userIdA").parquet('hdfs:/user/hl5679_nyu_edu/ml-latest/pairs_all')

    print("100 similarity pairs\n")
    similar_pairs = similar_pairs.filter("datasetA.userId < datasetB.userId").orderBy("JaccardDistance", ascending=True).limit(100)
    # top_100_pairs.select("datasetA.userId", "datasetB.userId", "JaccardDistance").show(100)
    # top_100_pairs.printSchema()

    print("Simplified df\n")
    simplified_df = similar_pairs.select(
        col("datasetA.userId").alias("userIdA"),
        col("datasetB.userId").alias("userIdB"),
        "JaccardDistance"
    )

    print("Write simplified df to CSV\n")
    # Write the simplified DataFrame to CSV
    simplified_df.write.partitionBy("userId").parquet('hdfs:/user/hl5679_nyu_edu/ml-latest/top_100_pairs_all')

    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('minHash').getOrCreate()

    spark = SparkSession.builder \
    .appName('minHash') \
    .config('spark.executor.memory', '4g') \
    .config('spark.driver.memory', '4g') \
    .config('spark.executor.instances', '10') \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "1") \
    .config("spark.dynamicAllocation.maxExecutors", "20") \
    .config("spark.shuffle.service.enabled", "true") \
    .getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)