import pickle
import json
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
ranforest=pickle.load(open('ranforest.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
#api
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=np.array(list(data.values())).reshape(1,-1)
    output=ranforest.predict(new_data)
    if output== 0:
        result = "Not Churn"
    else:
        result = "Churn"
    return jsonify({"Prediction": result})

if __name__=="__main__":
    app.run(debug=True)