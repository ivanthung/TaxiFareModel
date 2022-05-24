import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib

from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer:
    def __init__(self, X, y, **kwargs):
        """
        X: pandas DataFrame
        y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

        self.name = kwargs.get('name', 'undefined')
        self.user = kwargs.get('user', 'undefined')
        self.version = kwargs.get('version', '0.0')
        self.source = kwargs.get('source', 'undefined')
        self.experiment_name = kwargs.get('experiment_name', "[Nl] [AMS] [ivanthung] test_modle + v0.0")

    def set_pipeline(self, estimator):
        """returns a pipelined model"""
        dist_pipe = Pipeline(
            [("dist_trans", DistanceTransformer()), ("stdscaler", StandardScaler())]
        )
        time_pipe = Pipeline(
            [
                ("time_enc", TimeFeaturesEncoder("pickup_datetime")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
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
        pipe = Pipeline(
            [("preproc", preproc_pipe), ("linear_model", estimator)]
        )

        self.pipeline = pipe
        return self

    def train(self):
        """returns a trained pipelined model"""
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate_log(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)

        self.mlflow_log_metric("RSME", rmse)
        self.mlflow_log_param("Model", self.name)

        print(rmse)
        return rmse

    def save_model(self):
        """Save the trained model into a model.joblib file"""
        jobname = self.name + '.' + self.version + '.joblib'
        joblib.dump(self.pipeline, jobname)

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

    MLFLOW_URI = "https://mlflow.lewagon.ai/"

    # get data
    df = get_data()
    df_cleaned = clean_data(df)
    y = df_cleaned["fare_amount"]
    X = df_cleaned.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    models = {
        'Linear': LinearRegression(),
        'SGD Reg': SGDRegressor(),
        'Decision Tree': DecisionTreeRegressor()
    }

    for name, model in models.items():
        trainer = Trainer(X_train, y_train,
                          name=name,
                          version=1,
                          user='Ivanthung')

        trainer.set_pipeline(model)
        trainer.train()

        trainer.evaluate_log(X_val, y_val)
        result = cross_validate(trainer.pipeline, X, y,
                                cv = 5,
                                scoring = ['neg_root_mean_squared_error'],
                                n_jobs=-1
                                )['test_neg_root_mean_squared_error'].mean()*-1

        trainer.mlflow_log_metric("RSME-CV", result)

    # evaluate
