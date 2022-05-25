from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import AddGeohash, DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
