from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv("example_data/bank-additional-full.csv", sep=";")

# trans_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
#               "poutcome", "y"]

# for col in trans_cols:
#     lbe = LabelEncoder()
#     df[col] = lbe.fit_transform(df[col])

import xgboost as xgb
dtrain = xgb.DMatrix(data.drop("y", axis=1), label=data["y"])