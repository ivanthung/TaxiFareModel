import joblib
from termcolor import colored
import mlflow
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import (
    TimeFeaturesEncoder,
    DistanceTransformer,
    Optimizer,
    AddGeohash,
)
from TaxiFareModel.utils import compute_rmse, df_optimized
from TaxiFareModel.globalparams import (
    EXPERIMENT_NAME,
    BUCKET_NAME,
    MLFLOW_URI,
    STORAGE_LOCATION,
)
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from google.cloud import storage
import numpy as np


class Trainer(object):
    def __init__(self, nrows=1000):
        """
        X: pandas DataFrame
        y: pandas Series
        """
        self.df = get_data(nrows)
        self.pipeline = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME

    def clean_split(self, split=0.3):
        self.df = clean_data(self.df)
        self.df = df_optimized(self.df)
        self.y = self.df["fare_amount"]
        self.X = self.df.drop("fare_amount", axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=split
        )

    def set_experiment_name(self, experiment_name):
        """defines the experiment name for MLFlow"""
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline(
            [("dist_trans", DistanceTransformer()), ("stdscaler", StandardScaler())]
        )
        time_pipe = Pipeline(
            [
                ("time_enc", TimeFeaturesEncoder("pickup_datetime")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        geohash_pipe = make_pipeline(
            AddGeohash(precision=5), OneHotEncoder(handle_unknown="ignore", sparse=True)
        )

        preproc_pipe = ColumnTransformer(
            [
                (
                    "distance",
                    dist_pipe,
                    [
                        "pickup_latitude",
                        "pickup_longitude",
                        "dropoff_latitude",
                        "dropoff_longitude",
                    ],
                ),
                ("time", time_pipe, ["pickup_datetime"]),
            ],
            remainder="drop",
        )

        self.pipeline = Pipeline(
            [
                ("preproc", preproc_pipe),
                ("optimizer", Optimizer()),
                ("linear_model", LinearRegression()),
            ]
        )

    def fit(self):
        self.set_pipeline()
        self.mlflow_log_param("model", "Linear")
        self.pipeline.fit(self.X, self.y)

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        self.mlflow_log_metric("rmse", rmse)
        return round(rmse, 2)

    def upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename("model.joblib")

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, "model.joblib")
        print("saved model.joblib locally")

        # Implement here
        self.upload_model_to_gcp()
        print(
            f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}"
        )

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name
            ).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # Get and clean data
    # Train and save model, locally and
    params = dict(
        nrows=10000,
        upload=True,
        local=False,  # set to False to get data from GCP (Storage or BigQuery)
        gridsearch=False,
        optimize=True,
        estimator="xgboost",
        mlflow=True,  # set to True to log params to mlflow
        experiment_name=EXPERIMENT_NAME,
        pipeline_memory=None,  # None if no caching and True if caching expected
        distance_type="manhattan",
        feateng=[
            "distance_to_center",
            "direction",
            "distance",
            "time_features",
            "geohash",
        ],
        n_jobs=-1,
    )  # Try with njobs=1 and njobs = -1

    trainer = Trainer(nrows=1000)
    trainer.clean_split(split=0.3)
    trainer.set_experiment_name("xp2")
    trainer.fit()
    rmse = trainer.evaluate()

    print(f"rmse: {rmse}")
    # trainer.save_model()
