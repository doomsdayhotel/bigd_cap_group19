from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, avg, lit, expr
from evaluation import EvaluationUtil


class PopularityBaseline:
    def __init__(self, spark, df_train: DataFrame, df_test: DataFrame, top_n: int = 100):
        self.spark = spark
        self.df_train = df_train
        self.df_test = df_test
        self.top_n = top_n
        self.evaluation = EvaluationUtil(self.spark)

    def get_baseline(self, version='simple', dampening_factor=None):
        # Rank based on rating count
        if version == 'simple':
            baseline = self.df_train.groupBy('movieId').agg(
                count('rating').alias('rating_count')
            ).orderBy(col('rating_count').desc)

        # Rank weighted average using IMDb formula:
        # weighted rating = (rating_avg * rating_count + overall_avg * min_threshold)/(rating_count + min_threshold)
        elif version == 'weighted':
            # Average rating
            overall_avg = self.df_train.agg(
                avg('rating').alias('rating_overall_avg')
            ).collect()[0]['rating_overall_avg']

            # Minimum rating threshold
            min_threshold = self.df_train.approxQuantile('rating_count', [0.9], 0.05)[0]

            baseline = self.df_train.groupBy('movieId').agg(
                count('rating').alias('rating_count'),
                avg('rating').alias('rating_avg')
            ).withColumn(
                'weighted_average',
                (col('rating_avg') * col('rating_count') + lit(overall_avg) * lit(min_threshold)) /
                (col('rating_count') + lit(min_threshold))
            ).orderBy(col('weighted_average').desc())

        # Rank with dampening factor to balance rating
        elif version == 'dampened':
            if dampening_factor is None:
                dampening_factor = self.get_opt_dampening_factor()[0]
                print(f'Popularity baseline model with dampening_factor: {dampening_factor}')

            baseline = self.df_train.groupBy('movieId').agg(
                count('rating').alias('rating_count')
            ).withColumn(
                'rating_dampened', col('rating_count') / col('rating_count') + dampening_factor
            ).orderBy(col('rating_dampened').desc())

        else:
            raise ValueError("Unsupported version type specified for popularity baseline model")

        # Return top N popular movies
        return baseline.limit(self.top_n)

    def map_score(self, factor):
        df_pred = self.get_baseline(version='dampened', dampening_factor=factor)

        # Predictions
        y_pred = df_pred.groupBy('userId').agg(expr('collect_list(movieId) as pred_items'))

        # Ground truth
        y_true = self.df_test.groupBy('userId').agg(expr('collect_list(movieId) as true_items'))

        return factor, self.evaluation.compute_map_at_k(y_pred, y_true)

    def get_opt_dampening_factor(self, dampening_factors=None):
        # Set default parameters
        if dampening_factors is None:
            dampening_factors = [10, 50, 100, 200, 500, 1000]

        best_factor, best_map = max(map(self.map_score, dampening_factors), key=lambda x: x[1])
        return best_factor, best_map
