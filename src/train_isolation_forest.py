from preprocessing import load_and_preprocess_data
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import os

# Veri yükle
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/creditcard.csv")

# Isolation Forest modeli
iso_model = IsolationForest(n_estimators=100, contamination=0.0017, random_state=42)
iso_model.fit(X_train)

# Tahminleri yap (-1 = anomali, 1 = normal)
y_pred_train = iso_model.predict(X_train)
y_pred_test = iso_model.predict(X_test)

# Tahminleri 0/1 formatına çevir
y_pred_test = np.where(y_pred_test == -1, 1, 0)

# Raporları kaydet
report = classification_report(y_test, y_pred_test, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Kayıt klasörü
os.makedirs("reports", exist_ok=True)

# Raporları kaydet
pd.DataFrame(report).transpose().to_csv("reports/isolation_forest_report.csv")
pd.DataFrame(conf_matrix).to_csv("reports/isolation_forest_confusion_matrix.csv")

print(" Isolation Forest modeli eğitildi ve raporlar kaydedildi.")
