"""
Last Updated: Dec 12, 2018
Relative File Path: /main.py
Description: app routes
"""

from flask import Flask, render_template, request, redirect, jsonify
from wtforms import Form, TextAreaField, validators, TextField, TextAreaField, StringField, SubmitField
import os
import numpy as np

from sklearn.externals import joblib

import pickle


app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/results", methods=['POST'])
def predict_stuff():
    if request.method == 'POST':
        model = joblib.load('classifier_model.pkl')

        sex = int(request.form.get('sex'))
        maritalStatus = int(request.form.get('marital-status'))
        education_num = int(request.form.get('education-num'))
        age = int(request.form.get('age'))

        features_value =[[
        sex,
        maritalStatus,
        education_num,
        age,
        ]]

        predicted_income_values = model.predict(features_value)
        predicted_value = predicted_income_values[0]

        return render_template("results.html", pred="<= $50,000" if predicted_value < 0 else "> $50,000")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

