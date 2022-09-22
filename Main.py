import pandas as pd
from prepare_data import dropColumns, finalTreatment
from classifier import *


df = pd.read_csv('./Data/titanic_train.csv')

df = dropColumns(df)
df = finalTreatment(df)

X_train, X_test, y_train, y_test = getTrainTestVariables(df)
print(getPredictions(X_train, X_test, y_train, y_test))
