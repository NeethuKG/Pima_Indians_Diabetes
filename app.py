#import relevant libraries for flask, html rendering and loading the ML model
from flask import Flask,request, url_for, redirect, render_template
#from matplotlib import scale
import pickle
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
scale = joblib.load("scale-2.pkl")


@app.route("/")
def hello_world():
   return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():


   pregnancies = request.form['1']
   glucose = request.form['2']
   bloodPressure = request.form['3']
   skinThickness = request.form['4']
   insulin = request.form['5']
   bmi = request.form['6']
   dpf = request.form['7']
   age = request.form['8']

   rowDF= pd.DataFrame([pd.Series([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age])])
   print(rowDF)
   rowDF_new = pd.DataFrame(scale.transform(rowDF))




   return render_template('index.html')
  

if __name__ == '__main__':
   app.run(debug=True)






