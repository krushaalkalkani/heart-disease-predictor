import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('heart_attack_prediction_dataset.csv')
dataset.drop(['Patient ID', 'Country', 'Continent',
             'Hemisphere'], axis=1, inplace=True)

# Handle the Blood Pressure column
dataset['BP_Systolic'] = dataset['Blood Pressure'].str.split(
    '/').str[0].astype(int)
dataset['BP_Diastolic'] = dataset['Blood Pressure'].str.split(
    '/').str[1].astype(int)
dataset.drop('Blood Pressure', axis=1, inplace=True)

# Label encoding for sex
dataset['Sex'] = dataset['Sex'].map({'Male': 0, 'Female': 1})

# diet
dataset = pd.get_dummies(
    dataset, columns=['Diet'], drop_first=True, dtype=int, prefix='Diet')

# Separate features and target variable
X = dataset.drop('Heart Attack Risk', axis=1)
y = dataset['Heart Attack Risk']

# Split the dataset into training and testing sets
# it ensures reproducibility — the specific number is arbitrary
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler for future use
joblib.dump(sc, 'scaler.pkl')
