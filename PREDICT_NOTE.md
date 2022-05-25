
You can submit a prediction to **Kaggle** using:

``` python
python predict.py
```

⚠️ In order to push a submission to Kaggle, you need to use the whole dataset

You have two options:

## Download the test dataset from Kaggle

1. Download `test.csv` from [Kaggle](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data)

2. Save the file in the `TaxiFareModel` project as `data/test.csv`

3. Modify the call to `get_test_data` as follows in `generate_submission_csv`:

``` python
df_test = get_test_data(nrows, data="local")
```

## Use the full test dataset from S3

1. Modify the call to `get_test_data` as follows in `generate_submission_csv`:

``` python
df_test = get_test_data(nrows, data="full")
```
