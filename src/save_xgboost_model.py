import pickle
from xgboost import XGBClassifier
from preprocessing import load_and_preprocess_data

#  Veriyi yükle
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/creditcard.csv")

#  Modeli oluştur ve eğit
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

#  Kaydet
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" XGBoost modeli models/xgboost_model.pkl olarak kaydedildi.")
