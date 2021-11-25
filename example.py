from sklearn.preprocessing import LabelEncoder
import pandas as pd
import autogbm as ag

df = pd.read_csv("example_data/bank-additional-full.csv", sep=";")

trans_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
              "poutcome", "y"]

for col in trans_cols:
    lbe = LabelEncoder()
    df[col] = lbe.fit_transform(df[col])

ag.auto_train(train_set=df, label_name="y")
