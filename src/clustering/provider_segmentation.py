"""K-Means provider behavioral segmentation.

Aggregates claims to provider-level billing-behaviour features and segments
providers into behavioural clusters. The implementation uses PySpark MLlib
K-Means when a Spark session is available and transparently falls back to
scikit-learn K-Means otherwise, producing identical downstream columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.utils.logger import get_logger
from src.utils.spark_session import get_spark

logger = get_logger("provider-segmentation")


def build_provider_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate claims to one row per provider with billing-behaviour features."""
    g = df.groupby("provider_id")
    features = pd.DataFrame(
        {
            "avg_billed_amount": g["billed_amount"].mean(),
            "avg_billed_to_allowed_ratio": g["billed_to_allowed_ratio"].mean(),
            "avg_units": g["units"].mean(),
            "distinct_cpt_count": g["cpt_code"].nunique(),
            "distinct_icd_count": g["icd10_code"].nunique(),
            "claims_count": g.size(),
            "mismatch_rate": g["procedure_diagnosis_mismatch"].mean()
            if "procedure_diagnosis_mismatch" in df.columns else 0.0,
        }
    ).reset_index()
    return features


class ProviderSegmentation:
    def __init__(self, config=CONFIG.clustering) -> None:
        self.config = config
        self.model = None
        self.scaler: StandardScaler | None = None

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features = build_provider_features(df)
        feat_cols = list(self.config.feature_columns)
        n_clusters = min(self.config.n_clusters, max(2, len(features) // 2))

        spark = get_spark()
        if spark is not None:
            features = self._fit_predict_spark(spark, features, feat_cols, n_clusters)
        else:
            features = self._fit_predict_sklearn(features, feat_cols, n_clusters)

        logger.info(
            "Segmented %d providers into %d behavioural clusters",
            len(features), features["provider_cluster"].nunique(),
        )
        return features

    def _fit_predict_sklearn(self, features, feat_cols, n_clusters) -> pd.DataFrame:
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features[feat_cols].fillna(0.0))
        self.model = KMeans(n_clusters=n_clusters, random_state=self.config.random_state, n_init=10)
        features["provider_cluster"] = self.model.fit_predict(X)
        return features

    def _fit_predict_spark(self, spark, features, feat_cols, n_clusters) -> pd.DataFrame:
        from pyspark.ml.clustering import KMeans as SparkKMeans
        from pyspark.ml.feature import StandardScaler as SparkScaler
        from pyspark.ml.feature import VectorAssembler

        sdf = spark.createDataFrame(features[["provider_id"] + feat_cols].fillna(0.0))
        assembler = VectorAssembler(inputCols=feat_cols, outputCol="raw_features")
        sdf = assembler.transform(sdf)
        scaler = SparkScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
        sdf = scaler.fit(sdf).transform(sdf)
        km = SparkKMeans(k=n_clusters, seed=self.config.random_state, featuresCol="features",
                         predictionCol="provider_cluster")
        model = km.fit(sdf)
        result = model.transform(sdf).select("provider_id", "provider_cluster").toPandas()
        return features.merge(result, on="provider_id", how="left")
