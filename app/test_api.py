import requests
import random

# API URL'si
url = "http://localhost:5000/predict"

#  Rastgele 30 özellik oluştur (test amaçlı)
sample_features = [round(random.uniform(0, 1), 6) for _ in range(30)]

#  Kontrol amaçlı yazdır
print("📡 API'ye istek gönderiliyor...")
print("İstek verisi:", sample_features)

#  API'ye POST isteği gönder
response = requests.post(url, json={"features": sample_features})
print(" Yanıt alındı!")

#  Yanıtı göster
if response.status_code == 200:
    print(" Tahmin:", response.json())
else:
    print(" Hata:", response.status_code, response.text)

