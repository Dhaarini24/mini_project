from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
