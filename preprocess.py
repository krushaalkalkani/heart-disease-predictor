import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

dataset = pd.read_csv('heart_attack_prediction_dataset.csv')
dataset.drop(['Patient ID', 'Country', 'Continent',
             'Hemisphere'], axis=1, inplace=True)
# print(dataset.columns)

# Handle the Blood Pressure column
dataset['BP_Systolic'] = dataset['Blood Pressure'].str.split(
    '/').str[0].astype(int)
dataset['BP_Diastolic'] = dataset['Blood Pressure'].str.split(
    '/').str[1].astype(int)
dataset.drop('Blood Pressure', axis=1, inplace=True)

# print(dataset.head())

# Handling the categorical variables
print(dataset['Sex'].unique())
print(dataset['Diet'].unique())

# Label encoding for sex
dataset['Sex'] = dataset['Sex'].map({'Male': 0, 'Female': 1})

# diet
dataset = pd.get_dummies(
    dataset, columns=['Diet'], drop_first=True, dtype=int, prefix='Diet')

print(dataset.head())
print(dataset.dtypes)
