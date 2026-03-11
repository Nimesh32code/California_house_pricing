import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

# create flask app
app = Flask(__name__)

# load trained model
reg_model = pickle.load(open("regmodel.pkl", "rb"))

# load scaler
scaler = pickle.load(open("scaling.pkl", "rb"))


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction using API (Postman)
@app.route("/predict_api", methods=["POST"])
def predict_api():

    data = request.json
    print(data)

    new_data = np.array(list(data.values())).reshape(1, -1)

    new_data = scaler.transform(new_data)

    output = reg_model.predict(new_data)

    return jsonify({"output": output[0]})


# Prediction from Frontend Form
@app.route("/predict", methods=["POST"])
def predict():

    data = [float(x) for x in request.form.values()]

    final_input = scaler.transform(np.array(data).reshape(1, -1))

    output = reg_model.predict(final_input)[0]

    return render_template(
        "home.html",
        prediction_text="House Price Prediction is {}".format(output)
    )


# Run app
if __name__ == "__main__":
    app.run(debug=True)