import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Print request JSON data
        data = request.json['data']
        print("Received data:", data)
        
        # Convert data to numpy array
        data_array = np.array(list(data.values())).reshape(1, -1)
        print("Data array:", data_array)
        
        # Scale the data
        new_data = scaler.transform(data_array)
        print("Scaled data:", new_data)
        
        # Predict using the model
        output = regmodel.predict(new_data)
        print("Prediction:", output[0])
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': output[0]})
    except Exception as e:
        # Handle exceptions and return an error message
        print("Error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
