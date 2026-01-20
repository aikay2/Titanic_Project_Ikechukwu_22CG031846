from flask import Flask, render_template, request
import pickle
import numpy as np
import os  # <-- added to read PORT from environment

app = Flask(__name__)

with open("model/titanic_survival_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        pclass = int(request.form["pclass"])
        sex = int(request.form["sex"])
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        embarked = int(request.form["embarked"])

        data = np.array([[pclass, sex, age, fare, embarked]])
        data_scaled = scaler.transform(data)

        result = model.predict(data_scaled)[0]
        prediction = "Survived" if result == 1 else "Did Not Survive"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # <- Render sets this
    app.run(host="0.0.0.0", port=port, debug=True)
