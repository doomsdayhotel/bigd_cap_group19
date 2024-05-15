from pyspark.sql import DataFrame
from pyspark.sql.functions import expr
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics


class EvaluationUtil:
    def __init__(self, spark, k=100):
        self.spark = spark
        self.k = k

    def compute_map_at_k(self, y_pred: DataFrame, y_true: DataFrame) -> float:
        # Join ground truth with predictions on userId
        joined = y_pred.join(y_true, 'userId').rdd

        # Use map to calculate AP for each user
        map_score = joined.map(lambda row: self.calculate_apk(row['pred_items'], row['true_items'])).mean()

        return map_score

    @staticmethod
    def calculate_apk(pred_items, true_items):
        """
        Calculate Average Precision (AP) for a single user.

        :param pred_items: List of predicted items for the user.
        :param true_items: List of true items for the user.
        :return: AP score for the user.
        """
        if not true_items:
            return 0.0

        hits = [1 if p in true_items else 0 for p in pred_items]
        precisions = [sum(hits[:i + 1]) / (i + 1) for i in range(len(hits)) if hits[i] == 1]
        return sum(precisions) / len(true_items) if precisions else 0.0

    def evaluate(self, df_pred: DataFrame, df_true: DataFrame) -> dict:

        # Compute MAP
        y_pred = df_pred.groupBy('userId').agg(expr(f'collect_list(movieId)[:{self.k}] as pred_items'))
        y_true = df_true.groupBy('userId').agg(expr('collect_list(movieId) as true_items'))
        map_at_k = self.compute_map_at_k(y_pred, y_true)

        # Compute RMSE
        evaluator_rmse = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='rating')
        rmse_score = evaluator_rmse.evaluate(df_pred.join(df_true, ['userId', 'movieId']))

        # Prepare data for RankingMetrics
        rdd_pred = (df_pred.groupBy('userId').agg(expr('collect_list(movieId) as pred_items')).rdd
                    .map(lambda row: (row['userId'], row['pred_items'])))
        rdd_true = (df_true.groupBy('userId').agg(expr('collect_list(movieId) as true_items')).rdd
                    .map(lambda row: (row['userId'], row['true_items'])))
        combined_rdd = rdd_pred.join(rdd_true).map(lambda row: (row[1][0], row[1][1]))

        # Compute metrics using RankingMetrics
        metrics = RankingMetrics(combined_rdd)
        precision_at_k = metrics.precisionAt(self.k)
        ndcg_at_k = metrics.ndcgAt(self.k)
        mean_average_precision = metrics.meanAveragePrecision

        return {
            "K": self.k,
            "MAPAtK": map_at_k,
            "RMSE": rmse_score,
            "PrecisionAtK": precision_at_k,
            "NDCGAtK": ndcg_at_k,
            "MeanAveragePrecision": mean_average_precision # from RankingMetrics
        }
