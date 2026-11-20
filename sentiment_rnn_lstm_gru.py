"""
Comparative Study of SimpleRNN, LSTM, GRU and Bidirectional LSTM
for Sentiment Analysis on the IMDB Movie Review Dataset.

Run:  python sentiment_rnn_lstm_gru.py
All plots / tables are saved to the ./results folder.
"""
import os, time, json, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ------------------------------------------------------------------ config
SEED = 42
VOCAB_SIZE = 10000      # keep the 10k most frequent words
MAX_LEN = 200           # pad / truncate every review to 200 words
EMBED_DIM = 64
UNITS = 64
EPOCHS = 5
BATCH = 128
OUT = "results"

os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ------------------------------------------------------------------ data
print("Loading IMDB dataset ...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
print(f"Train: {len(x_train)} reviews | Test: {len(x_test)} reviews")
lengths = [len(s) for s in x_train]
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="pre", truncating="pre")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="pre", truncating="pre")

# review-length distribution plot (dataset analysis figure)
plt.figure(figsize=(7, 4))
plt.hist(lengths, bins=60, color="#4C72B0")
plt.axvline(MAX_LEN, color="red", ls="--", label=f"MAX_LEN = {MAX_LEN}")
plt.xlim(0, 1000); plt.xlabel("Review length (words)"); plt.ylabel("Count")
plt.title("IMDB review length distribution"); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/fig0_review_lengths.png", dpi=150); plt.close()

# ------------------------------------------------------------------ models
def build(kind):
    m = models.Sequential(name=kind)
    m.add(layers.Input(shape=(MAX_LEN,)))
    m.add(layers.Embedding(VOCAB_SIZE, EMBED_DIM))
    if kind == "SimpleRNN":
        m.add(layers.SimpleRNN(UNITS))
    elif kind == "LSTM":
        m.add(layers.LSTM(UNITS))
    elif kind == "GRU":
        m.add(layers.GRU(UNITS))
    elif kind == "BiLSTM":
        m.add(layers.Bidirectional(layers.LSTM(UNITS)))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

MODEL_NAMES = ["SimpleRNN", "LSTM", "GRU", "BiLSTM"]
histories, rows, cms, trained = {}, [], {}, {}

for name in MODEL_NAMES:
    print(f"\n================ Training {name} ================")
    tf.random.set_seed(SEED)
    model = build(name)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    t0 = time.time()
    h = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                  validation_split=0.2, callbacks=[es], verbose=2)
    train_time = time.time() - t0

    t0 = time.time()
    prob = model.predict(x_test, batch_size=512, verbose=0).ravel()
    infer_time = time.time() - t0
    pred = (prob >= 0.5).astype(int)

    rows.append({
        "Model": name,
        "Parameters": model.count_params(),
        "Test Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "Epochs run": len(h.history["loss"]),
        "Train time (s)": round(train_time, 1),
        "Inference time (s)": round(infer_time, 2),
    })
    histories[name] = h.history
    cms[name] = confusion_matrix(y_test, pred)
    trained[name] = model
    print(classification_report(y_test, pred, target_names=["Negative", "Positive"]))

# ------------------------------------------------------------------ tables
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results_table.csv", index=False)
with open(f"{OUT}/histories.json", "w") as f:
    json.dump(histories, f, indent=1)
print("\n\nFINAL RESULTS\n", df.to_string(index=False))

# ------------------------------------------------------------------ figures
colors = dict(zip(MODEL_NAMES, ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for n in MODEL_NAMES:
    e = range(1, len(histories[n]["accuracy"]) + 1)
    ax[0].plot(e, histories[n]["accuracy"], "--", color=colors[n], alpha=.6)
    ax[0].plot(e, histories[n]["val_accuracy"], "-o", color=colors[n], label=n)
    ax[1].plot(e, histories[n]["loss"], "--", color=colors[n], alpha=.6)
    ax[1].plot(e, histories[n]["val_loss"], "-o", color=colors[n], label=n)
ax[0].set_title("Accuracy (solid = validation, dashed = train)")
ax[1].set_title("Loss (solid = validation, dashed = train)")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_training_curves.png", dpi=150); plt.close()

metrics = ["Test Accuracy", "Precision", "Recall", "F1-score"]
plot_df = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
plt.figure(figsize=(10, 5))
b = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette=colors)
for c in b.containers: b.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
plt.ylim(0.4, 1.0); plt.title("Test-set performance comparison"); plt.tight_layout()
plt.savefig(f"{OUT}/fig2_metric_comparison.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
for a, n in zip(ax, MODEL_NAMES):
    sns.heatmap(cms[n], annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    a.set_title(n); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_confusion_matrices.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(df["Model"], df["Train time (s)"], color=[colors[n] for n in df["Model"]])
ax[0].set_title("Training time (s)")
ax[1].bar(df["Model"], df["Parameters"], color=[colors[n] for n in df["Model"]])
ax[1].set_title("Number of trainable parameters")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_time_and_params.png", dpi=150); plt.close()

# ------------------------------------------------------------------ live demo
word_index = imdb.get_word_index()
def encode(text):
    ids = [1]  # start token
    for w in text.lower().replace(",", " ").replace(".", " ").replace("!", " ").split():
        i = word_index.get(w, -1) + 3
        ids.append(i if 3 <= i < VOCAB_SIZE else 2)  # 2 = unknown
    return pad_sequences([ids], maxlen=MAX_LEN)

samples = [
    "This movie was absolutely wonderful, the acting was brilliant and the story was touching",
    "Worst film I have ever seen. Boring, badly acted and a complete waste of time",
    "The plot was predictable but the music and visuals were great",
    "I did not like this movie at all, it was not good",
]
best = df.sort_values("Test Accuracy", ascending=False).iloc[0]["Model"]
demo = []
for s in samples:
    r = {"Review": s}
    for n in MODEL_NAMES:
        p = float(trained[n].predict(encode(s), verbose=0)[0, 0])
        r[n] = f"{'POS' if p >= .5 else 'NEG'} ({p:.2f})"
    demo.append(r)
demo_df = pd.DataFrame(demo)
demo_df.to_csv(f"{OUT}/demo_predictions.csv", index=False)
print("\nCUSTOM REVIEW PREDICTIONS\n", demo_df.to_string(index=False))
print(f"\nBest model: {best}.  All outputs saved in ./{OUT}/")
