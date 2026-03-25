import pandas as pd

dataset = pd.read_csv('heart_attack_prediction_dataset.csv')
dataset.drop(['Patient ID', 'Country', 'Continent',
             'Hemisphere'], axis=1, inplace=True)
print(dataset.columns)
