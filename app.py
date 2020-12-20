from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pickle



app = Flask(__name__)


sc_X = pickle.load(open('scalar.pkl','rb'))
classifier = pickle.load(open('classifier.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getname',methods=['POST','GET'])
def get_name():
    if request.method =='POST':
        name = "Prerana"
        text = request.form['lastname']   
        names = ['Kush','Nancy','Amit']
        len1 = len(names)
    return render_template('index.html', surname=text,name=name,length=len1,friends=names)

@app.route('/effort',methods=['POST','GET'])
def effort():
    if request.method=='POST':
        b = request.form['bug']
        r = request.form['rfc']

        x = [b,r]
        x = np.array(x)
        x = x.reshape(1,-1)
       
        scaled_x = sc_X.transform(x)
    
        Y_Pred = classifier.predict(scaled_x)

    return render_template('prediction.html', prediction=Y_Pred)
    
    #return redirect(url_for('index')) # if you donot want to render any data


if __name__ == "__main__":
    app.run()