from preprocessing import load_and_preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import os

#  Veri yükle
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/creditcard.csv")

#  Sadece normal (Class=0) verilerle autoencoder eğitilir
X_train_normal = X_train[y_train == 0]

#  Autoencoder mimarisi
input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

#  Eğitim
autoencoder.fit(X_train_normal, X_train_normal,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_split=0.1,
                verbose=1)

#  Rekonstrüksiyon hataları
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - reconstructions), axis=1)

#  Eşik belirle (IQR yöntemi)
threshold = np.percentile(mse, 95)

#  Anomali tespiti
y_pred = (mse > threshold).astype(int)

#  Raporlar
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, mse)

#  Kayıt
os.makedirs("reports", exist_ok=True)
pd.DataFrame(report).transpose().to_csv("reports/autoencoder_report.csv")
pd.DataFrame(conf_matrix).to_csv("reports/autoencoder_confusion_matrix.csv")
with open("reports/autoencoder_auc.txt", "w") as f:
    f.write(f"ROC-AUC: {roc_auc:.4f}\nThreshold: {threshold:.6f}")

print(" Autoencoder eğitildi ve sonuçlar kaydedildi.")
