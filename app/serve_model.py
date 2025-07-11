from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

#  Flask başlat
app = Flask(__name__)
CORS(app)

#  Modeli yükle
with open("models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Anomalik XGBoost API çalışıyor."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")
        
        if not features or len(features) != 30:
            return jsonify({"error": "Exactly 30 features must be provided."}), 400

        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        return jsonify({"prediction": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#  Uygulamayı başlat
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
