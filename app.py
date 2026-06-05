from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load(
    "models/fake_news_model.pkl"
)

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    vector = vectorizer.transform([news])

    result = model.predict(vector)[0]

    if result == 0:
        prediction = "FAKE NEWS"
    else:
        prediction = "REAL NEWS"

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)