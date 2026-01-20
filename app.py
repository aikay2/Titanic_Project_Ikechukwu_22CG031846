from flask import Flask, render_template, request
import pickle
import numpy as np

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
    app.run(debug=True)
