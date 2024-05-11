#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ spark-submit --deploy-mode client q3_partition.py <file_path>
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, rand, row_number
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

    # Add a row number column to each user's ratings to facilitate splitting
    window_spec = Window.partitionBy("userid")
    ratings = ratings.withColumn("row_number", row_number().over(window_spec))

    # Calculate the total number of ratings per user
    user_total_ratings = ratings.groupBy("userid").agg(count("row_number").alias("total_ratings"))

    # Join back to include total ratings in the DataFrame
    ratings = ratings.join(user_total_ratings, on="userid", how="inner")

    # Calculate split indices
    split_expr = col("row_number") / col("total_ratings")
    
    # Define conditions for each split
    train_ratings = ratings.filter(split_expr <= 0.6)
    val_ratings = ratings.filter((split_expr > 0.6) & (split_expr <= 0.8))
    test_ratings = ratings.filter(split_expr > 0.8)

    # Save partitioned data for future use
    train_ratings.write.csv("hdfs:/user/qy561_nyu_edu/ml-latest-small/train_ratings.csv", header=True)
    val_ratings.write.csv("hdfs:/user/qy561_nyu_edu/ml-latest-small/val_ratings", header=True)
    test_ratings.write.csv("hdfs:/user/qy561_nyu_edu/ml-latest-small/test_ratings.csv", header=True)

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
    file_path = f'/hdfs:/user/{userID}/ml-latest-small/ratings.csv'
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
