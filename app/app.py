

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load(r"/Users/geethuvishnu/Downloads/AIDI2004_lAB4/app/model.pkl")  # Load your trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request content-type must be application/json'}), 400
    
    data = request.json.get('data')
    # Ensure the input data is a list of the correct length
    if not isinstance(data, list) or len(data) != 6:
        return jsonify({'error': 'Input data must be a list of 6 features.'}), 400

    # Convert the input data to a numpy array
    final_input = np.array(data).reshape(1, -1)
    
    # Predict using the loaded model
    try:
        prediction = model.predict(final_input)[0]
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
