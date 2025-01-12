#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client q3_partition.py <file_path>
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, rand, row_number, when
from pyspark.sql.window import Window

def partition(spark, file_path):
    """
    Partition the data by user into training, validation, and test sets.
    """
    # Load the ratings data
    ratings = spark.read.csv(file_path, header=True, inferSchema=True)

    # # Define the minimum number of ratings required per user
    # min_ratings = 15

    # # Filter users with at least 'min_ratings' ratings
    # user_rating_counts = ratings_df.groupBy("userid").agg(count("rating").alias("rating_count"))
    # sufficient_ratings_users = user_rating_counts.filter(col("rating_count") >= min_ratings)ß

    # Join back to get the filtered ratings data
    # filtered_ratings_df = ratings_df.join(sufficient_ratings_users, on="userid", how="inner")

    # Define the window specification by user and order by a random value
    window_spec = Window.partitionBy("userId").orderBy(rand())

    # Add a row number to each user's ratings to facilitate partitioning
    ratings = ratings.withColumn("row_num", row_number().over(window_spec))

    # Calculate the number of ratings per user
    user_rating_counts = ratings.groupBy("userId").agg(count("rating").alias("rating_count"))

    # Join the counts back to the ratings dataframe
    ratings = ratings.join(user_rating_counts, on="userId", how="inner")

    # Define the splitting conditions
    ratings = ratings.withColumn(
        "set",
        when(col("row_num") <= col("rating_count") * 0.7, "train")
        .when(col("row_num") <= col("rating_count") * 0.85, "validation")
        .otherwise("test")
    )

    # Split the data into training, validation, and test sets
    train_ratings = ratings.filter(col("set") == "train").drop("row_num", "rating_count", "set")
    val_ratings = ratings.filter(col("set") == "validation").drop("row_num", "rating_count", "set")
    test_ratings = ratings.filter(col("set") == "test").drop("row_num", "rating_count", "set")

    # Save partitioned data for future use
    train_ratings.write.mode('overwrite').csv("hdfs:/user/qy561_nyu_edu/ml-latest-small/train_ratings.csv", header=True)
    val_ratings.write.mode('overwrite').csv("hdfs:/user/qy561_nyu_edu/ml-latest-small/val_ratings.csv", header=True)
    test_ratings.write.mode('overwrite').csv("hdfs:/user/qy561_nyu_edu/ml-latest-small/test_ratings.csv", header=True)

    # Display the count of records in each set
    print(f"Training data count: {train_ratings.count()}")
    print(f"Validation data count: {val_ratings.count()}")
    print(f"Test data count: {test_ratings.count()}")


def main(spark, userID):
    """
    Main function to execute the data processing and partitioning.
    
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    """
    '''1. Preprocessing Data '''
    # Load the ratings.csv into DataFrame
    file_path = f'hdfs:///user/{userID}/ml-latest-small/ratings.csv'
    # f'/hdfs:/user/{userID}/ml-latest-small/ratings.csv'
    partition(spark, file_path)

# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('q3_partition').getOrCreate()

    # Get userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
