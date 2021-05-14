from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('stroke_regression.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])

        avg_glucose_level = float(request.form['avg_glucose_level'])

        bmi = float(request.form['bmi'])

        hypertension = int(request.form['hypertension'])

        heart_disease = int(request.form['heart_disease'])
        
        #conditions for gender
        gender = request.form['gender']
        if(gender=='Male'):
                gender_Male=1
                gender_Other=0
        elif(gender=='Other'):
            gender_Male=0
            gender_Other=1
        else:
            gender_Male=0
            gender_Other=0
        
        #conditions for marriage status
        residence = request.form['residence']
        if(residence=='Urban'):
            Residence_type_Urban=1
        else:
            Residence_type_Urban=0

        #conditions for residence
        married = request.form['married']
        if(married=='Yes'):
            ever_married_Yes=1
        else:
            ever_married_Yes=0

        #conditions for work type
        work = request.form['work_type']
        if(work=='Private'):
            work_type_Never_worked=0
            work_type_Private=1
            work_type_Self_employed=0
            work_type_children=0
        elif(work=='Self_employed'):
            work_type_Never_worked=0
            work_type_Private=0
            work_type_Self_employed=1
            work_type_children=0
        elif(work=='Never_worked'):
            work_type_Never_worked=1
            work_type_Private=0
            work_type_Self_employed=0
            work_type_children=0
        elif(work=='children'):
            work_type_Never_worked=0
            work_type_Private=0
            work_type_Self_employed=0
            work_type_children=1
        else:
            work_type_Never_worked=0
            work_type_Private=0
            work_type_Self_employed=0
            work_type_children=0

        #conditions for smoke status
        smoke = request.form['smoke_status']
        if(smoke=='formerly_smoked'):
            smoking_status_formerly_smoked=1
            smoking_status_never_smoked=0
            smoking_status_smokes=0
        elif(smoke=='never_smoked'):
            smoking_status_formerly_smoked=0
            smoking_status_never_smoked=1
            smoking_status_smokes=0
        elif(smoke=='smokes'):
            smoking_status_formerly_smoked=0
            smoking_status_never_smoked=0
            smoking_status_smokes=1
        else:
            smoking_status_formerly_smoked=0
            smoking_status_never_smoked=0
            smoking_status_smokes=0

        prediction=model.predict([[age,hypertension,heart_disease,avg_glucose_level,bmi,gender_Male,gender_Other,ever_married_Yes,work_type_Never_worked,work_type_Private,work_type_Self_employed,work_type_children,Residence_type_Urban,smoking_status_formerly_smoked,smoking_status_never_smoked,smoking_status_smokes]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_text="Sorry you have entered invalid details")
        else:
            if(output==1):
                return render_template('index.html',prediction_text="You have a chance of stroke".format(output))
            else:
                return render_template('index.html',prediction_text="You have no or less chance of stroke".format(output))

    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)