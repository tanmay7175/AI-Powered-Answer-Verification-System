from flask import Flask, request, render_template
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load AI Model and Preprocessing Objects
model = load_model("model/text_classification_cnn.h5")

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Predefined correct answer (you can change this)
CORRECT_ANSWER = "The capital of France is Paris."

@app.route("/", methods=["GET", "POST"])
def home():
    prediction, confidence = None, None
    if request.method == "POST":
        bot_message = request.form["bot_message"]
        user_response = request.form["user_response"]
        correct_answer = CORRECT_ANSWER  # Hide this from frontend

        # Combine input text
        combined_text = f"Bot: {bot_message}\nAnswer: {correct_answer}\nStudent's Response: {user_response}"

        # Preprocess input for model
        vectorized_text = vectorizer.transform([combined_text]).toarray()
        vectorized_text = np.expand_dims(vectorized_text, axis=-1)

        # Predict classification
        pred_probs = model.predict(vectorized_text)
        predicted_label = label_encoder.inverse_transform([np.argmax(pred_probs)])[0]
        confidence = float(np.max(pred_probs))

        return render_template("home.html", prediction=predicted_label, confidence=confidence, correct_answer=correct_answer)

    return render_template("home.html", correct_answer=CORRECT_ANSWER)

if __name__ == "__main__":
    app.run(debug=True)
