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
    ratings_df = spark.read.csv(f'hdfs:/user/{userID}/ml-latest-small/ratings.csv', schema='movieId STRING, title STRING, genres STRING')

    # Get all unique movieIds
    # unique_movie_ids = ratings_df.select("movieId").distinct().rdd.flatMap(lambda x: x).collect()
    # total_movies = movies_df.agg(max("movieId")).collect()[0][0]
    # total_movies = total_movies + 1
    # print(total_movies) #193609

    
    # Group by userId and collect all movieIds into a list
    ratings_df_grouped = ratings_df.groupBy("userId").agg(collect_list("movieId").alias("movieIds"))
    # Show the transformed DataFrame
    ratings_df_grouped.show()
    # ratings_df_grouped.write.csv('hdfs:/user/hl5679_nyu_edu/ml-latest-small/ratings_df_grouped.csv', header=True, mode="overwrite")
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
    ratings_df_final = cv.fit(ratings_df_grouped)
    ratings_df_final.show()
    # ratings_df_final.write.csv('hdfs:/user/hl5679_nyu_edu/ml-latest-small/ratings_df_final.csv', header=True, mode="overwrite")
    '''
    +------+--------------------+--------------------+
    |userId|            movieIds|            features|
    +------+--------------------+--------------------+
    |   148|[356, 1197, 4308,...|(193610,[356,1197...|
    |   463|[110, 296, 356, 5...|(193610,[110,296,...|
    |   471|[1, 296, 356, 527...|(193610,[1,296,35...|
    +------+--------------------+--------------------+
    '''

    ''' 2. Applying MinHash '''
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=10)
    model = mh.fit(ratings_df_final)


    transformed_df = model.transform(ratings_df_final)
    similar_pairs = model.approxSimilarityJoin(transformed_df, transformed_df, 0.6, distCol="JaccardDistance")

    similar_pairs = similar_pairs.filter("datasetA.userId < datasetB.userId")  # Avoid duplicates and self-pairs
    sorted_pairs = similar_pairs.orderBy("JaccardDistance", ascending=True)
    top_100_pairs = sorted_pairs.limit(100)
    # top_100_pairs.select("datasetA.userId", "datasetB.userId", "JaccardDistance").show(100)
    # top_100_pairs.printSchema()

    simplified_df = top_100_pairs.select(
        col("datasetA.userId").alias("userIdA"),
        col("datasetB.userId").alias("userIdB"),
        "JaccardDistance"
    )
    # Write the simplified DataFrame to CSV
    simplified_df.write.csv('hdfs:/user/hl5679_nyu_edu/ml-latest-small/top_100_pairs2.csv', header=True, mode="overwrite")

    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('minHash').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)


''' Error message
24/05/09 19:22:30 INFO org.sparkproject.jetty.server.AbstractConnector: Stopped Spark@71e6cb53{HTTP/1.1, (http/1.1)}{0.0.0.0:0}
24/05/09 19:23:00 WARN org.apache.hadoop.util.ShutdownHookManager: ShutdownHook '$anon$2' timeout, java.util.concurrent.TimeoutException
java.util.concurrent.TimeoutException
        at java.util.concurrent.FutureTask.get(FutureTask.java:205)
        at org.apache.hadoop.util.ShutdownHookManager.executeShutdown(ShutdownHookManager.java:124)
        at org.apache.hadoop.util.ShutdownHookManager$1.run(ShutdownHookManager.java:95)
24/05/09 19:23:00 WARN org.apache.spark.SparkContext: Ignoring Exception while stopping SparkContext from shutdown hook
java.lang.InterruptedException
        at java.lang.Object.wait(Native Method)
        at java.lang.Thread.join(Thread.java:1257)
        at java.lang.Thread.join(Thread.java:1331)
        at org.apache.spark.scheduler.AsyncEventQueue.stop(AsyncEventQueue.scala:148)
        at org.apache.spark.scheduler.LiveListenerBus.$anonfun$stop$1(LiveListenerBus.scala:269)
        at org.apache.spark.scheduler.LiveListenerBus.$anonfun$stop$1$adapted(LiveListenerBus.scala:269)
        at scala.collection.Iterator.foreach(Iterator.scala:943)
        at scala.collection.Iterator.foreach$(Iterator.scala:943)
        at scala.collection.AbstractIterator.foreach(Iterator.scala:1431)
        at scala.collection.IterableLike.foreach(IterableLike.scala:74)
        at scala.collection.IterableLike.foreach$(IterableLike.scala:73)
        at scala.collection.AbstractIterable.foreach(Iterable.scala:56)
        at org.apache.spark.scheduler.LiveListenerBus.stop(LiveListenerBus.scala:269)
        at org.apache.spark.SparkContext.$anonfun$stop$13(SparkContext.scala:2083)
        at org.apache.spark.util.Utils$.tryLogNonFatalError(Utils.scala:1419)
        at org.apache.spark.SparkContext.stop(SparkContext.scala:2082)
        at org.apache.spark.SparkContext.$anonfun$new$37(SparkContext.scala:669)
        at org.apache.spark.util.SparkShutdownHook.run(ShutdownHookManager.scala:214)
        at org.apache.spark.util.SparkShutdownHookManager.$anonfun$runAll$2(ShutdownHookManager.scala:188)
        at scala.runtime.java8.JFunction0$mcV$sp.apply(JFunction0$mcV$sp.java:23)
        at org.apache.spark.util.Utils$.logUncaughtExceptions(Utils.scala:1996)
        at org.apache.spark.util.SparkShutdownHookManager.$anonfun$runAll$1(ShutdownHookManager.scala:188)
        at scala.runtime.java8.JFunction0$mcV$sp.apply(JFunction0$mcV$sp.java:23)
        at scala.util.Try$.apply(Try.scala:213)
        at org.apache.spark.util.SparkShutdownHookManager.runAll(ShutdownHookManager.scala:188)
        at org.apache.spark.util.SparkShutdownHookManager$$anon$2.run(ShutdownHookManager.scala:178)
        at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511)
        at java.util.concurrent.FutureTask.run(FutureTask.java:266)
        at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
        at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
        at java.lang.Thread.run(Thread.java:750)

'''