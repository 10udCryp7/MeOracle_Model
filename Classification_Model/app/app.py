from flask import Flask, request, jsonify

import torch
import data_setup
import engine
import model
import io

app = Flask(__name__)

MODEL_PATH = "model/model_latest.pth"

@app.route('/predict', methods=['POST'])

def predict():
    # Get the symptoms from the request
    symptoms = request.json['symptoms']
    num = request.json['num']
    # Transform the symptoms into a tensor
    input_data = data_setup.transform(symptoms)
    # Predict
    output = engine.predict(input_data, num, MODEL_PATH)
    result = {
        "output": output.tolist()
    }
    # Return the result
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




