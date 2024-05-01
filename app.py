from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
app = Flask(__name__)

class SoilFertilityPredictor:
    def __init__(self):
        self.data = None
        self.X = None
        self.Y = None
        self.scaler = MinMaxScaler()
        self.forestRegressor = RandomForestRegressor(criterion='squared_error', max_depth=8, n_estimators=10, random_state=0)

    def load_data(self,processed_data_set):
        self.data = pd.read_csv(r"processed_data_set.csv")
        self.X, self.Y = self.data[self.data.columns[1:]], self.data['Vegetation Cover']
        self.preprocess_data()
        self.train_model()

    def preprocess_data(self):
        X_scaled = self.scaler.fit_transform(self.X.values)
        self.X = X_scaled[:-1]
        self.Y = self.Y[:-1]

    def train_model(self, test_size=0.1, random_state=43):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)
        self.forestRegressor.fit(X_train, Y_train)
        self.calculate_accuracy(X_test, Y_test)

    def predict_fertility(self, input_data):
        prediction = self.forestRegressor.predict([input_data])
        return prediction[0]

    def evaluate_fertility(self, prediction, nutrient_values):
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = nutrient_values
        text = ""
        if prediction < 90:
            if a0 < 12.75 or a2 < 47 or a8 < 0.6 or a3 < 15 or a6 < 0.28 or a10 < 1:
                text = "Your Soil is less fertile. You may try increasing these nutrients: "
                if a0 < 12.75:
                    text += "NO3 "
                if a2 < 47:
                    text += "P "
                if a8 < 0.6:
                    text += "Zn "
                if a3 < 15:
                    text += "K "
                if a6 < 0.28:
                    text += "Organic Matter "
                if a10 < 1:
                    text += "Fe "
        elif prediction >= 90:
            text = "Your soil is highly fertile."
        return text

    def calculate_accuracy(self, X_test, Y_test):
        accuracy = self.forestRegressor.score(X_test, Y_test)
        print(f"Model Accuracy: {accuracy}")

predictor = SoilFertilityPredictor()
predictor.load_data(r"dataset.txt")
# Load the pre-trained model
with open('new_rf_model.pickle', 'rb') as f:
    model = pickle.load(f)

class_labels = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
                'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
                'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
                'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

@app.route('/')
def index():
    return render_template('landing.html', header='templates\header.html', footer='templates/footer.html')

@app.route('/contact')
def contact():
    return render_template('contact_us.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')
@app.route('/suggestion')
def suggestion():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.Series({
        'N': data['N'],
        'P': data['P'],
        'K': data['K'],
        'temperature': data['temperature'],
        'humidity': data['humidity'],
        'ph': data['ph'],
        'rainfall': data['rainfall']
    })
    prediction = model.predict([input_data])[0]
    return jsonify({'crop': class_labels[prediction]})

@app.route('/soil_pred')
def soil_pred():
    return render_template('soil_pred.html')

@app.route('/predict_fertility', methods=['POST'])
def predict_fertility():
    data = request.form  # Assuming form data is being sent from index.html
    nutrient_values = [
        float(data['NO3']),
        float(data['NH4']),
        float(data['P']),
        float(data['K']),
        float(data['SO4']),
        float(data['B']),
        float(data['Organic_Matter']),
        float(data['pH']),
        float(data['Zn']),
        float(data['Cu']),
        float(data['Fe']),
        float(data['Ca']),
        float(data['Mg']),
        float(data['Na'])
    ]

    prediction = predictor.predict_fertility(nutrient_values)
    result_text = predictor.evaluate_fertility(prediction, nutrient_values)
    prediction_percentage = int(round(prediction))
    return render_template('result.html', prediction=prediction_percentage,result_text=result_text, prediction_text=prediction_percentage)
if __name__ == '__main__':
    app.run(debug=True)
