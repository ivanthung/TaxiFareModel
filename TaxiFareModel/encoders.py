from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from scipy.sparse import csr_matrix
import pygeohash as gh

from TaxiFareModel.utils import haversine_vectorized, df_optimized


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
    Extracts the day of week (dow), the hour, the month and the year from a time column.
    Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self, time_column, time_zone_name="America/New_York"):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[["dow", "hour", "month", "year"]]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Computes the haversine distance between two GPS points.
    Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(
        self,
        start_lat="pickup_latitude",
        start_lon="pickup_longitude",
        end_lat="dropoff_latitude",
        end_lon="dropoff_longitude",
    ):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = haversine_vectorized(
            X_,
            start_lat=self.start_lat,
            start_lon=self.start_lon,
            end_lat=self.end_lat,
            end_lon=self.end_lon,
        )
        return X_[["distance"]]


class Optimizer(BaseEstimator, TransformerMixin):
    """
    Optimzes the dataframe and returns
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X.toarray())
        df_ = df.copy()
        df_ = df_optimized(df_)
        return csr_matrix(df_.values)

class AddGeohash(BaseEstimator, TransformerMixin):
    """
    Add a geohash (ex: "dr5rx") of len "precision" = 5 by default
    corresponding to each (lon,lat) tuple, for pick-up, and drop-off
    """

    def __init__(self, precision=5):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["geohash_pickup"] = X_.apply(
            lambda x: gh.encode(
                x.pickup_latitude, x.pickup_longitude, precision=self.precision
            ),
            axis=1,
        )
        X_["geohash_dropoff"] = X_.apply(
            lambda x: gh.encode(
                x.dropoff_latitude, x.dropoff_longitude, precision=self.precision
            ),
            axis=1,
        )
        return X_[["geohash_pickup", "geohash_dropoff"]]
