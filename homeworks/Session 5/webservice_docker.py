
from flask import Flask
from flask import request
from flask import jsonify

import pickle

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


app = Flask('Webservice into Docker')

def loading_model():
    # Loading model
    model_file = './model2.bin'
    with open(model_file, 'rb') as f_model_in:
        model = pickle.load(f_model_in)
    
    # Loading dv
    dv_file = './dv.bin'
    with open(dv_file, 'rb') as f_dv_in:
        dv = pickle.load(f_dv_in)
    return model, dv

@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()    
    model, dv = loading_model()
    y_pred = predict_single(customer, dv, model)
    credit_card = y_pred >= 0.5
    
    result = {"credit_card_probability": float(y_pred),
            "credit_card": bool(credit_card)}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=9696)