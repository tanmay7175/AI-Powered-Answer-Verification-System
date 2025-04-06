from flask import Flask, request, render_template
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load Model and Preprocessing Objects
model = load_model("model/text_classification_cnn.h5")

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction, confidence = None, None
    if request.method == "POST":
        bot_message = request.form["bot_message"]
        correct_answer = request.form["correct_answer"]
        user_response = request.form["user_response"]

        # Combine text
        combined_text = f"Bot: {bot_message}\nAnswer: {correct_answer}\nStudent's Response: {user_response}"

        # Preprocess input
        vectorized_text = vectorizer.transform([combined_text]).toarray()
        vectorized_text = np.expand_dims(vectorized_text, axis=-1)

        # Predict
        pred_probs = model.predict(vectorized_text)
        predicted_label = label_encoder.inverse_transform([np.argmax(pred_probs)])[0]
        confidence = float(np.max(pred_probs))

        return render_template("home.html", prediction=predicted_label, confidence=confidence)

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
