
from preprocessing import load_and_preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import xgboost as xgb
import numpy as np
import os


X_train, X_test, y_train, y_test = load_and_preprocess_data("data/creditcard.csv")


model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)


os.makedirs("reports", exist_ok=True)
pd.DataFrame(report).transpose().to_csv("reports/xgboost_report.csv")
pd.DataFrame(conf_matrix).to_csv("reports/xgboost_confusion_matrix.csv")
with open("reports/xgboost_auc.txt", "w") as f:
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")

print(" XGBoost modeli eğitildi ve sonuçlar kaydedildi.")