from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        Geo = request.form['Geo']
        LP = request.form['LP']
        Features_SAT_Value = int(request.form['Features_SAT_Value'])
        Features_SAT_Design=int(request.form['Features_SAT_Design'])
        Features_SAT_Weight = int(request.form['Features_SAT_Weight'])
        Features_SAT_Performance = int(request.form['Features_SAT_Performance'])
        Features_SAT_Quality = int(request.form['Features_SAT_Quality'])
        Equipment_SAT_OS = int(request.form['Equipment_SAT_OS'])
        Equipment_SAT_Keyboard = int(request.form['Equipment_SAT_Keyboard'])
        Equipment_SAT_Noise = int(request.form['Equipment_SAT_Noise'])
        Equipment_SAT_Battery= int(request.form['Equipment_SAT_Battery'])
        Brazil = 0
        EMEA = 0
        LAS = 0
        NA = 0
        Other = 0
        if (Geo == 'Brazil'):
            Brazil = 1
        elif (Geo=="EMEA"):
            EMEA = 1
        elif (Geo=="LAS"):
            LAS = 1
        elif (Geo=="NA"):
            NA = 1
        else:
            Other = 1
        if(LP=="Consumer"):
            SMB=0
        else:
            SMB=1
        Predict = model.predict([[Brazil, EMEA, LAS, NA, Other, SMB,Features_SAT_Value, Features_SAT_Design, Features_SAT_Weight,Features_SAT_Performance, Features_SAT_Quality, Equipment_SAT_OS,Equipment_SAT_Keyboard, Equipment_SAT_Noise,Equipment_SAT_Battery]])
        output = round(Predict[0], 0)
        if output < 0:
            return render_template('index.html', prediction_texts="some error in the entered value")
        else:
            return render_template('index.html', prediction_text="NPS rating {}".format(output))
    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)