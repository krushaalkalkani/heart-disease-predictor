import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Traing logistic model on the training set
lm_classifier = LogisticRegression(max_iter=1000, random_state=42)
lm_classifier.fit(X_train, y_train)

# Evaluating the model on the test set
y_pred_lm = lm_classifier.predict(X_test)
print(classification_report(y_test, y_pred_lm))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_lm))

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluating the Random Forest with the test set
y_pred_rf = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))

# Train the XGBoost Classifier
xgb_classifier = XGBClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_classifier.fit(X_train, y_train)

# Evaluating the XGBoost model with the test set
y_pred_xgb = xgb_classifier.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_xgb))
