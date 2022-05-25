import joblib
import TaxiFareModel.featureng as fng
from termcolor import colored
import mlflow
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import Optimizer
from TaxiFareModel.utils import compute_rmse, df_optimized
from TaxiFareModel.globalparams import (
    EXPERIMENT_NAME,
    BUCKET_NAME,
    MLFLOW_URI,
    STORAGE_LOCATION,
)
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from google.cloud import storage
import numpy as np


class Trainer(object):
    def __init__(self, **kwargs):
        """
        X: pandas DataFrame
        y: pandas Series
        """
        self.upload = kwargs.get("upload", False)
        self.local = kwargs.get("local", True)
        self.nrows = kwargs.get("nrows", 1000)
        self.feateng = kwargs.get('feateng', ['distance', 'time'])
        self.gridsearch = kwargs.get('gridsearch', False)
        self.optimize = kwargs.get('optimize', True)
        self.estimator = kwargs.get('estimator', 'linear')
        self.mlflow = kwargs.get('mlflow', False),  # set to True to log params to mlflow
        self.experiment_name = EXPERIMENT_NAME
        self.pipeline_memory = kwargs.get('pipeline_memory', True)
        self.distance_type = kwargs.get("distance_type", 'manhattan')
        self.n_jobs = kwargs.get('n_jobs', -1)

        self.df = get_data(self.nrows, self.local)
        self.pipeline = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def clean_split(self, split=0.3):
        self.df = clean_data(self.df)
        self.df = df_optimized(self.df) if self.optimize else self.df
        self.y = self.df["fare_amount"]
        self.X = self.df.drop("fare_amount", axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=split
        )

    def set_experiment_name(self, experiment_name):
        """defines the experiment name for MLFlow"""
        self.experiment_name = experiment_name

    def create_estimator(self):
        """ Creates estimator and implements grid search for params if requested"""
        estimators = {
            'linear' : LinearRegression(),
            'xgboost': XGBRegressor(),
            'decisiontree' : DecisionTreeRegressor(),
            'sgdregresseor' : SGDRegressor()
            }

        estimator = estimators.get(self.estimator)
        estimator.set_params(n_jobs=self.n_jobs)
        print(f"..setting up {self.estimator} as estimator")
        return estimator

    def set_pipeline(self):
        """Defines the pipeline as a class attribute, takes params from self"""

        prepro_dc = {
            'distance' : (fng.dist_pipe, ["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]),
            'time' : (fng.time_pipe, ["pickup_datetime"]),
            'geohash' : (fng.geohash_pipe, ["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"])
        }

        feat_eng = [(feat,) + prepro_dc[feat] for feat in self.feateng if feat in prepro_dc.keys()]
        preproc_pipe = ColumnTransformer(feat_eng, remainder="drop")

        self.pipeline = Pipeline(
            [
                ("preproc", preproc_pipe),
                ("optimizer", Optimizer()) if self.optimize else ('', None),
                (self.estimator, self.create_estimator()),
            ], memory=self.pipeline_memory
        )

    def fit(self):
        self.set_pipeline()
        self.mlflow_log_param("model", self.estimator)
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
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, "model.joblib")
        print("saved model.joblib locally")

        # Implement here
        if self.upload:
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
        nrows=1000,
        upload=False,
        local=True,  # set to False to get data from GCP (Storage or BigQuery)
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
            "time",
            "geohash"
        ],
        n_jobs=-1,
    )  # Try with njobs=1 and njobs = -1

    trainer = Trainer(**params)
    trainer.clean_split(split=0.3)
    trainer.fit()
    rmse = trainer.evaluate()
    trainer.save_model()

    print(f"rmse: {rmse}")
