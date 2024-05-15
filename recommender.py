import argparse
import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, explode
from baseline import PopularityBaseline
from evaluation import EvaluationUtil


class Recommender:
    def __init__(self, spark: SparkSession, file_base_path: str, k: int = 100):
        self.spark = spark
        self.k = k
        self.datasets = self.load_data(['train', 'val', 'test'], file_base_path)

        self.train_df = self.datasets['train']
        self.val_df = self.datasets['val']
        self.test_df = self.datasets['test']

        self.baseline_model = PopularityBaseline(spark, self.train_df, self.val_df, self.k)
        self.evaluator = EvaluationUtil(spark, self.k)

        self.als_model = None

    def load_data(self, keys, file_base_path):
        return {key: self.spark.read.format('parquet').load(f'{file_base_path}/{key}.parquet') for key in keys}

    def train_baseline(self):
        """
        Train the baseline recommender model.
        """
        self.poularity_baseline = self.baseline_model.get_baseline(version='weighted')

    def train_als(self):
        als = ALS(userCol='userId',
                  itemCol='movieId',
                  ratingCol='rating',
                  coldStartStrategy='drop',
                  nonnegative=True)

        param_grid = (ParamGridBuilder()
                      .addGrid(als.rank, [10, 20, 30])
                      .addGrid(als.maxIter, [10, 15, 20])
                      .addGrid(als.regParam, [0.1, 0.01, 0.001])
                      .build())

        # Use CrossValidator to find the best model
        cv = CrossValidator(estimator=als,
                            estimatorParamMaps=param_grid,
                            evaluator=RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol="prediction"),
                            numFolds=3,
                            collectSubModels=True)

        models = cv.fit(self.train_df)

        # Evaluate each model using MAP
        best_model = None
        best_map = -1  # Initializing to a very low value
        for model in models.subModels:
            top_n_recs = self.predict_top_n(model)
            map_val = self.evaluator.compute_map_at_k(top_n_recs, self.val_df.select('userId', 'movieId'))
            if map_val > best_map:
                best_map = map_val
                best_model = model

        self.als_model = best_model

    def evaluate(self):
        """
        Evaluate the recommender models.
        """
        # Evaluate the baseline recommendations
        baseline_metrics = self.evaluator.evaluate(
            df_pred=self.poularity_baseline.select('userId', 'movieId', 'rating'),
            df_true=self.test_df.select('userId', 'movieId', 'rating'))

        # Evaluate ALS model
        als_predictions = self.als_model.transform(self.test_df)
        als_metrics = self.evaluator.evaluate(
            df_pred=als_predictions.select('userId', 'movieId', 'rating'),
            df_true=self.test_df.select('userId', 'movieId', 'rating'))

        return {
            "Baseline Metrics": baseline_metrics,
            "ALS Metrics": als_metrics
        }

    def recommend_for_user(self, user_id: int, model_type: str = 'als'):
        """
        Recommend top K items for a given user using the specified model.
        """
        if model_type == 'baseline':
            user_recommendations = self.poularity_baseline.filter(col('userId') == user_id)
        elif model_type == 'als':
            user_recommendations = self.als_model.recommendForUserSubset(
                self.spark.createDataFrame([(user_id,)], ["userId"]), self.k
            ).select("userId", "recommendations.movieId")
        else:
            raise ValueError("Unsupported model type. Use 'baseline' or 'als'.")

        return user_recommendations

    def predict_top_n(self, model):
        """
        Get top N recommendations for each user.
        """
        user_recs = model.recommendForAllUsers(self.k)

        user_recs = user_recs.select("userId", explode("recommendations").alias("rec"))
        user_recs = user_recs.select("userId", col("rec.movieId").alias("movieId"), col("rec.rating").alias("rating"))
        return user_recs


def main(spark, userID, args):
    file_base_path = f'hdfs:///user/{userID}/{args.data_dir}'

    recommender = Recommender(spark, file_base_path, k=args.top_n)

    # Cache data
    recommender.train_df.cache()
    recommender.val_df.cache()
    recommender.test_df.cache()

    evaluation_util = EvaluationUtil(spark)

    # Popularity baseline model
    baseline = PopularityBaseline(spark, recommender.train_df, recommender.val_df, args.top_n)
    evaluator = EvaluationUtil(spark, args.top_n)

    # Popularity baselines
    popularity_baseline_simple = baseline.get_baseline(version='simple')
    popularity_baseline_simple.show(20)

    print(
        'Evaluation for simple baseline model:',
        evaluator.evaluate(
            df_pred=popularity_baseline_simple.select('userId', 'movieId', 'rating'),
            df_true=recommender.val_df.select('userId', 'movieId', 'rating')))

    # Popularity baselines
    popularity_baseline_weighted = baseline.get_baseline(version='weighted')
    popularity_baseline_weighted.show(20)

    print(
        'Evaluation for weighted baseline model:',
        evaluator.evaluate(
            df_pred=popularity_baseline_weighted.select('userId', 'movieId', 'rating'),
            df_true=recommender.val_df.select('userId', 'movieId', 'rating')))

    # Popularity baselines
    popularity_baseline_dampened = baseline.get_baseline(version='dampened')
    popularity_baseline_dampened.show(20)

    print(
        'Evaluation for dampened baseline model:',
        evaluator.evaluate(
            df_pred=popularity_baseline_dampened.select('userId', 'movieId', 'rating'),
            df_true=recommender.val_df.select('userId', 'movieId', 'rating')))

    # Train ALS model
    recommender.train_als()
    best_model = recommender.als_model
    best_rmse = best_model.avgMetrics[0]  # Average RMSE across the folds
    print(f'Best ALS model RMSE on validation set: {best_rmse}')

    # Predict top N items for test set using ALS model
    top_n_als_df = recommender.predict_top_n(best_model)

    # Evaluate ALS model on test set
    map_test_als = evaluation_util.compute_map_at_k(top_n_als_df, recommender.test_df.select('userId', 'movieId'))
    print(f'ALS Model Test Set MAP: {map_test_als}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n', type=int, default=100, help='Number of top N items')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing input data')
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName('Movie Recommender') \
        .config("spark.sql.shuffle.partitions", "800") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.2") \
        .getOrCreate()

    userID = os.environ['USER']
    main(spark, userID, args)
    spark.stop()
