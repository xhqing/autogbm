import pandas as pd
import numpy as np

solution = pd.read_csv("madeline.solution", sep=" ", header=None)

print(solution.columns)
print(solution.shape)
print(solution.head())

y_test = pd.DataFrame(np.load("madeline.data/test_npy/y_test.npy"))

print(y_test.columns)
print(y_test.shape)
print(y_test.head())
print(y_test.loc[:,0].unique())
print(y_test.loc[:,1].unique())

y_train = pd.DataFrame(np.load("madeline.data/train_npy/y_train.npy"))
print(y_train.loc[:,0].unique())
print(y_train.loc[:,1].unique())

