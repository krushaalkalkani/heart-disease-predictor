import numpy as np
import pandas as pd
import joblib
from sklearn.base import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score
from joblib import dump


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
    n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluating the Random Forest with the test set
y_pred_rf = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))

# Train the XGBoost Classifier
xgb_classifier = XGBClassifier(
    n_estimators=100, random_state=42, learning_rate=0.1, eval_metric='logloss')
xgb_classifier.fit(X_train, y_train)

# Evaluating the XGBoost model with the test set
y_pred_xgb = xgb_classifier.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_xgb))

# Comparing the models based on their performance metrics
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [accuracy_score(y_test, y_pred_lm), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_xgb)],
    'Recall': [classification_report(y_test, y_pred_lm, output_dict=True)['1']['recall'],
               classification_report(y_test, y_pred_rf, output_dict=True)[
        '1']['recall'],
        classification_report(y_test, y_pred_xgb, output_dict=True)['1']['recall']],
    'F1': [classification_report(y_test, y_pred_lm, output_dict=True)['1']['f1-score'],
           classification_report(y_test, y_pred_rf, output_dict=True)[
        '1']['f1-score'],
        classification_report(y_test, y_pred_xgb, output_dict=True)['1']['f1-score']],
    'ROC-AUC': [roc_auc_score(y_test, y_pred_lm), roc_auc_score(y_test, y_pred_rf), roc_auc_score(y_test, y_pred_xgb)]

})

print(results.sort_values('Recall', ascending=False))

# save the best model
joblib.dump(best_model, 'model.pkl')
print("Best model saved!")
