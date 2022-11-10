import pickle
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify


model_file = "finalmodel"

with open(model_file, "rb") as f:
    model = pickle.load(f)

app = Flask("Fraud")

@app.route("/predict", methods = ["POST"])
def predict():
    offer = request.get_json()

    X = pd.DataFrame([offer])
    y_pred = model.predict_proba(X)[0, 1]
    fraud = y_pred >= 0.5

    result = {
        "fraud_probability": float(y_pred), 
        "fraud": bool(fraud)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host = "0.0.0.0", port=9696)
