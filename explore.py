# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('heart_attack_prediction_dataset.csv')


print(dataset.head(5))
print(dataset.shape)

print("-- Data Types --")
print(dataset.dtypes)

print("-- Missing Values --")
print(dataset.isnull().sum())

print("-- Summary Statistics --")
print(dataset.describe())

print("--Target Distribution--")
print(dataset['Heart Attack Risk'].value_counts())

# Create a countplot of the target variable (Heart Attack Risk) — this shows the imbalance visually
sns.countplot(x='Heart Attack Risk', data=dataset)

# Create a histogram of Age — to see the age distribution of patients
sns.histplot(dataset['Age'], bins=20, kde=True)

# Create a correlation heatmap of all numeric columns — this shows which features are related to each other
numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 8))
sns.heatmap(dataset[numeric_cols].corr(), annot=True,
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()
