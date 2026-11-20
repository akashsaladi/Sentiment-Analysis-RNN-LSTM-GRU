"""
Live demo: type any movie review and the trained GRU model predicts Positive / Negative.

Run:   python 1_code/predict.py                       (interactive - type reviews, 'q' to quit)
       python 1_code/predict.py "the movie was great"  (one review from the command line)

First run only: if 2_results/model/gru_demo.keras does not exist, it trains the GRU model
(about 2-3 minutes on CPU) and saves it. After that it loads instantly.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE, MAX_LEN = 10000, 200
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder
MODEL_PATH = os.path.join(BASE, "2_results", "model", "gru_demo.keras")

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("No saved model found - training the GRU model once (2-3 minutes)...")
    tf.random.set_seed(42); np.random.seed(42)
    (x_tr, y_tr), _ = imdb.load_data(num_words=VOCAB_SIZE)
    x_tr = pad_sequences(x_tr, maxlen=MAX_LEN)
    model = models.Sequential([layers.Input(shape=(MAX_LEN,)),
                               layers.Embedding(VOCAB_SIZE, 64), layers.GRU(64),
                               layers.Dropout(0.5), layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_tr, y_tr, epochs=2, batch_size=128, validation_split=0.2, verbose=2)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print("Saved", MODEL_PATH)

word_index = imdb.get_word_index()

def predict(text):
    ids = [1]
    for w in "".join(c if c.isalnum() or c == "'" else " " for c in text.lower()).split():
        i = word_index.get(w, -1) + 3
        ids.append(i if 3 <= i < VOCAB_SIZE else 2)
    p = float(model.predict(pad_sequences([ids], maxlen=MAX_LEN), verbose=0)[0, 0])
    label = "POSITIVE 😊" if p >= 0.5 else "NEGATIVE 😠"
    print(f"  -> {label}   (probability of positive = {p:.2f})\n")

if len(sys.argv) > 1:
    predict(" ".join(sys.argv[1:]))
else:
    print("\nType a movie review and press Enter (type q to quit)\n")
    while True:
        t = input("Review: ").strip()
        if t.lower() in ("q", "quit", "exit"):
            break
        if t:
            predict(t)
