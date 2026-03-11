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


# home page
@app.route("/")
def home():
    return render_template("home.html")


# prediction API
@app.route("/predict_api", methods=["POST"])
def predict_api():

    data = request.json
    print(data)

    # convert values to list
    new_data = np.array(list(data.values())).reshape(1, -1)

    # scale input
    new_data = scaler.transform(new_data)

    # prediction
    output = reg_model.predict(new_data)

    return jsonify({"output": output[0]})


# run app
if __name__ == "__main__":
    app.run(debug=True)