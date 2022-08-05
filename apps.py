from modelwithacc import Prediction_Function
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify,render_template
import pickle
app = Flask(__name__)


model = pickle.load(open('modelwithacc.pkl','rb'))

@app.route('/',methods=["GET","POST"])
def main():
    if request.method == 'GET':
        return render_template("index.html")
    
    if request.method == 'POST':
        cate = request.form['cate']
        date = request.form['date']
        #date = flask.request.form['date']
       #result = model(cate,date)
        prediction = model(cate,date)
        output = prediction
        #output=[1,2]
        
        return render_template("index.html", prediction_text='Number of orders predicted is :{}'.format(output))
    

    
    



if __name__ == '__main__':
    app.run(port=8000, debug=True)