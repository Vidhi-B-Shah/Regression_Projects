import pickle
from flask import Flask, request, render_template, send_from_directory
import numpy as np
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Import your pipeline classes here
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            country=request.form.get('country'),
            state=request.form.get('state'),
            city=request.form.get('city'),
            station=request.form.get('station'),
            pollutant_min=float(request.form.get('pollutant_min')),
            pollutant_max=float(request.form.get('pollutant_max'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results[0])

# Route to serve the background image
@app.route('/static/air.webp')
def serve_static(filename):
    root_dir = Path(__file__).parent
    return send_from_directory(root_dir / 'static', filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0")