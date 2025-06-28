import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load test dataset
def load_test_data(desc_path, solution_path):
    descriptions = []
    genres = []
    with open(desc_path, "r", encoding="utf-8") as desc_file, open(solution_path, "r", encoding="utf-8") as sol_file:
        for desc_line, sol_line in zip(desc_file, sol_file):
            desc_parts = desc_line.strip().split(" ::: ")
            sol_parts = sol_line.strip().split(" ::: ")
            if len(desc_parts) >= 3 and len(sol_parts) >= 4:
                descriptions.append(desc_parts[2])  # Extract description
                genres.append(sol_parts[2])  # Extract genre
    return pd.DataFrame(zip(descriptions, genres), columns=["plot", "genre"])

# Paths to test dataset
test_data_path = "test_data.txt"
test_solution_path = "test_data_solution.txt"

# Load test data
df_test = load_test_data(test_data_path, test_solution_path)
X_test, y_test = df_test["plot"], df_test["genre"]

# Function to evaluate a saved model
def evaluate_saved_model(model_name):
    # Load trained model and vectorizer
    try:
        vectorizer, model = joblib.load(f"{model_name}_model.pkl")
    except ValueError:
        print(f"Error loading {model_name}_model.pkl. Ensure it was saved correctly.")
        return

    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Predict and evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Model: {model_name}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}\n")

# Evaluate each model
evaluate_saved_model("logistic_regression")
evaluate_saved_model("random_forest")
evaluate_saved_model("knn")

