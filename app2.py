from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Function to predict genre
def predict_genre(description, model_name):
    vectorizer, model = joblib.load(f'{model_name}_model.pkl')
    description_tfidf = vectorizer.transform([description])
    return model.predict(description_tfidf)[0]

@app.route("/", methods=["GET", "POST"])
def index():
    description = ""
    predicted_genre = None
    selected_model = "logistic_regression"

    if request.method == "POST":
        description = request.form["description"]
        selected_model = request.form["model"]
        predicted_genre = predict_genre(description, selected_model)

    return render_template("index.html", description=description, predicted_genre=predicted_genre, selected_model=selected_model)

if __name__ == "__main__":
    app.run(debug=True)
