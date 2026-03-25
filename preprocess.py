import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# print(dataset.head())
# print(dataset.dtypes)

# Separate features and target variable
X = dataset.drop('Heart Attack Risk', axis=1).values
y = dataset['Heart Attack Risk'].values

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    # "it ensures reproducibility — the specific number is arbitrary
    X, y, test_size=0.2, random_state=42)


# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train.shape)
print(X_test.shape)
