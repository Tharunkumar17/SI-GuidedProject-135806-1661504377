import pandas as pd
from flask import Flask,render_template,request
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pickle

app = Flask(__name__)
model = pickle.load(open("ckd.pkl", 'rb'))

@app.route("/", methods=['GET'])
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        Blood_pressure = float(request.form["Blood_pressure"])
        Specafic_gravity = float(request.form['Specafic_gravity'])
        albumin = float(request.form["albumin"])
        red_blood_cells = float(request.form["red_blood_cells"])
        pus_cell = float(request.form["pus_cell"])
        pus_cell_clumps = float(request.form["pus_cell_clumps"])
        blood_glucose_random = float(request.form["blood_glucose_random"])
        blood_urea = float(request.form["blood_urea"])
        serum_creatinine = float(request.form["serum_creatinine"])
        sodium = float(request.form["sodium"])

        hemoglobin = float(request.form["hemoglobin"])
        packed_cell_volume = float(request.form["packed_cell_volume"])
        red_blood_cell_count = float(request.form["red_blood_cell_count"])
        hypertension = float(request.form["hypertension"])
        diabetesmellitus = float(request.form["diabetesmellitus"])
        coronary_artery_disease = float(request.form["coronary_artery_disease"])
        appetite = float(request.form["appetite"])
        pedal_edema = float(request.form["pedal_edema"])
        
        values = np.array([[Blood_pressure, Specafic_gravity, albumin, red_blood_cells, pus_cell, pus_cell_clumps, blood_glucose_random, blood_urea, serum_creatinine, sodium, hemoglobin, packed_cell_volume, red_blood_cell_count, hypertension, diabetesmellitus, coronary_artery_disease, appetite, pedal_edema]])
        
        prediction = model.predict(values)
        
        return render_template("result.html", prediction=prediction)
    
if __name__ == "__main__":
    app.run(debug=True)