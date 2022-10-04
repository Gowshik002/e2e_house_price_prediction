import pickle

import numpy as np
from flask import Flask,  render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('desktop.ini')


@app.route('/predict', methods=['POST', 'GET'])
def results():
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])
    sqft_living = float(request.form['sqft_living'])
    sqft_lot = float(request.form['sqft_lot'])
    floors = float(request.form['floors'])
    waterfront = float(request.form['waterfront'])
    view = float(request.form['view'])
    condition = float(request.form['condition'])
    grade = float(request.form['grade'])
    sqft_above = float(request.form['sqft_above'])
    sqft_basement = float(request.form['sqft_basement'])
    sqft_living15 = float(request.form['sqft_living15'])
    sqft_lot15 = float(request.form['sqft_lot15'])

    x = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, sqft_living15, sqft_lot15]])
    model = pickle.load(open('model.pkl', 'rb'))
    y_prediction = model.predict(x)
    return jsonify({'Model Prediction': float(y_prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=1010)
