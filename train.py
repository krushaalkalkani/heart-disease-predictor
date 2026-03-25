import numpy as np
from sklearn.linear_model import LogisticRegression

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Traing logistic model on the training set
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)
