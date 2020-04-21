from flask import Flask, jsonify, request
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
from flask_cors import CORS

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)

tokenizer = None
with open(os.path.relpath("bin/tokenizer.pickle"), "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model(os.path.relpath("bin/model.h5"))


def clean_text(text):
    tokens = text.split(" ")
    tokens = [token.lower() for token in tokens if token not in stop_words]
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if not re.search("^@", token)
        and not re.search("^http", token)
        and not re.search("^\:", token)
        and not re.search("^\;", token)
    ]
    new_text = " ".join(tokens)
    return "".join(ch for ch in new_text if ch not in string.punctuation)


def make_prediction(text):
    pad_type = "post"
    trunc_type = "post"
    max_length = 100
    cleaned_text = clean_text(text)
    text_sequence = tokenizer.texts_to_sequences(np.array([cleaned_text]))
    padded_sequence = pad_sequences(
        text_sequence, maxlen=max_length, padding=pad_type, truncating=trunc_type
    )
    prediction = model.predict(padded_sequence)
    return prediction


@app.route("/", methods=["POST"])
def index():
    data = request.get_json()
    prediction = make_prediction(data["text"])[0][0]
    return {
        "text": data["text"],
        "prediction": {
            "positive": prediction.astype(float),
            "negative": 1 - prediction.astype(float),
        },
    }


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 33507)))
