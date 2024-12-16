# flask_server.py
from flask import Flask, request, jsonify
from pyngrok import ngrok, conf
import pickle
from joblib import load
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

# Izinkan semua CORS
CORS(app)

# Load model dari file .pkl
with open('Cloud Computing Stunting Detection\model.pkl', 'rb') as f:
    model1 = pickle.load(f)

model2 = load('Cloud Computing Stunting Detection\diabetes_model_noscale.pkl')


conf.get_default().auth_token = "2onWfrfzKiKgYC3qfzpTXz2RH7f_Fzc4xvLjaKhvDc5WZu4S"

# Endpoint untuk prediksi
@app.route('/predict-stunting', methods=['POST'])
def predict1():
    input_data = request.get_json()
    features = input_data['features']

    # Validasi input
    if len(features) != 3:
        return jsonify({'error': 'Input must have 3 features'}), 400

    # Membuat prediksi
    input_array = np.array([features])
    prediction = model1.predict(input_array)
    return jsonify({'prediction': int(prediction[0])})

@app.route('/predict-diabetes', methods=['POST'])
def predict2():
    input_data = request.get_json()
    features = input_data['features']

    # Validasi input
    if len(features) != 8:
        return jsonify({'error': 'Input must have 8 features'}), 400

    # Membuat prediksi
    input_array = np.array([features])
    prediction = model2.predict(input_array)
    return jsonify({'prediction': int(prediction[0])})

@app.route('/calculate-bmi', methods=['POST'])
def calculate_bmi():
    input_data = request.get_json()
    weight = input_data.get('weight')
    height = input_data.get('height')

    # Validasi input
    if weight is None or height is None:
        return jsonify({'error': 'Weight and height are required'}), 400
    if not isinstance(weight, (int, float)) or not isinstance(height, (int, float)):
        return jsonify({'error': 'Weight and height must be numbers'}), 400
    if weight <= 0 or height <= 0:
        return jsonify({'error': 'Weight and height must be greater than zero'}), 400

    # Menghitung BMI
    bmi = weight / (height ** 2)

    # Klasifikasi BMI
    if bmi < 18.5:
        classification = 'Underweight'
    elif 18.5 <= bmi < 24.9:
        classification = 'Normal weight'
    elif 25 <= bmi < 29.9:
        classification = 'Overweight'
    else:
        classification = 'Obese'

    return jsonify({'bmi': round(bmi, 2), 'classification': classification})

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel URL: {public_url}")
    app.run(port=5000)
