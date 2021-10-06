from sklearn.preprocessing import LabelEncoder
from autotab.autotab import autotab_run
import pandas as pd

df = pd.read_csv("example_data/bank-additional-full.csv", sep=";")

trans_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
              "poutcome", "y"]

for col in trans_cols:
    lbe = LabelEncoder()
    df[col] = lbe.fit_transform(df[col])

label = df["y"]

autotab_run(df=df, label=label, formatted_dir="./formatted_data")
