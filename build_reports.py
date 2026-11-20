"""
Fills the college template (deep_learning_project_report.docx) with the project content
and writes the same report as LaTeX (Report_LaTeX/report.tex).

Run:  python build_reports.py
Out:  Final_Project_Report.docx, Report_LaTeX/ (report.tex + figures/)
"""
import copy, os, re, shutil, zipfile
import pandas as pd
import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn

TEMPLATE = "deep_learning_project_report.docx"
OUT_DOCX = "Final_Project_Report.docx"
TEX_DIR = "Report_LaTeX"
R = "results"

TITLE = ("A Comparative Study of RNN, LSTM, GRU and BiLSTM Models for "
         "Sentiment Analysis of IMDB Movie Reviews")
STUDENTS = [("HARSHA SIVA AKASH SALADI", "AP23110011490"),
            ("KISHAN BHIMAVARAPU", "AP23110011351")]
GUIDE, GUIDE_DESIG = "Dr. Lalit Kumar", "Assistant Professor, Department of CSE"
MONTH = "November, 2026"

df = pd.read_csv(f"{R}/results_table.csv")
demo = pd.read_csv(f"{R}/demo_predictions.csv")
g = {r["Model"]: r for _, r in df.iterrows()}
acc = lambda m: f"{g[m]['Test Accuracy'] * 100:.2f}%"

# =====================================================================  CONTENT
# block types: h1, h2, p, bullets, fig, table, code, eq
C = []
h1 = lambda t: C.append(("h1", t))
h2 = lambda t: C.append(("h2", t))
p = lambda t: C.append(("p", t))
bl = lambda items: C.append(("bullets", items))
fig = lambda f, cap, w=6.0: C.append(("fig", f, cap, w))
tab = lambda head, rows, cap: C.append(("table", head, rows, cap))
code = lambda t: C.append(("code", t))
eq = lambda tex, plain: C.append(("eq", tex, plain))

h1("Abstract")
p("Sentiment analysis is a Natural Language Processing (NLP) task that automatically classifies the "
  "opinion expressed in a piece of text as positive or negative. Because text is sequential data, in "
  "which word order changes meaning, recurrent neural networks (RNNs) are a natural choice for this task. "
  "However, several recurrent architectures exist, and it is not obvious which one gives the best balance "
  "between accuracy and computational cost. This project presents a controlled comparison of four "
  "recurrent architectures, namely SimpleRNN, Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU) "
  "and Bidirectional LSTM (BiLSTM), on the IMDB movie review dataset of 50,000 reviews. All models use "
  "identical preprocessing (10,000-word vocabulary, sequences of 200 tokens), a 64-dimensional word "
  "embedding, 64 recurrent units, dropout and the same training settings, so only the recurrent layer "
  "differs. The models are evaluated on 25,000 unseen test reviews using accuracy, precision, recall, "
  "F1-score, confusion matrices, training time and parameter count. "
  f"The gated models clearly outperform the SimpleRNN ({acc('SimpleRNN')}): LSTM achieves the highest "
  f"test accuracy of {acc('LSTM')}, BiLSTM the highest F1-score of {g['BiLSTM']['F1-score']:.3f}, and "
  f"GRU reaches {acc('GRU')} with fewer parameters and less training time. BiLSTM needs about 1.8 times "
  "the training time of LSTM without an accuracy gain. The results confirm that gating mechanisms reduce "
  "the vanishing gradient problem, and that LSTM or GRU offer the best trade-off for practical sentiment "
  "classification.")
p("Keywords: Sentiment Analysis, Recurrent Neural Network, LSTM, GRU, Bidirectional LSTM, IMDB, Deep Learning, NLP.")

# --------------------------------------------------------------- 1
h1("1. Introduction")
h2("1.1 Background")
p("Every day, users publish millions of product reviews, movie reviews, tweets and comments. This text "
  "contains valuable information about public opinion, but reading it manually is impossible. Sentiment "
  "analysis, also called opinion mining, is the NLP task of automatically deciding whether a text "
  "expresses a positive or a negative opinion. It is used by e-commerce platforms such as Amazon and "
  "Flipkart to summarise product reviews, by companies to monitor their brand on social media, by "
  "customer-support teams to prioritise complaints, and by movie platforms such as IMDB to measure "
  "audience reaction.")
p("The main difficulty is that the meaning of a sentence depends on the order and context of its words. "
  "For example, \"not good\" has the opposite meaning of \"good\", and in \"the plot was predictable but "
  "the music was great\" the clause after \"but\" decides the overall sentiment. Traditional bag-of-words "
  "models and feed-forward neural networks ignore this order.")
h2("1.2 Significance and Context")
p("Recurrent Neural Networks (RNNs) process text one word at a time and keep a hidden state that acts as "
  "a memory of the words seen so far, which makes them suitable for sequential data. However, the basic "
  "RNN suffers from the vanishing gradient problem, so it forgets information from the beginning of long "
  "sequences. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks add gates that "
  "control the flow of information, and bidirectional networks read the sequence in both directions. "
  "These architectures are core topics of the Deep Learning course (Recurrent Neural Networks, Unfolded "
  "RNNs, Seq2Seq, LSTM, GRU and Encoder-Decoder architectures), and this project studies them "
  "experimentally on a real-world task.")
h2("1.3 Problem Statement")
p("Given a movie review written in English, classify it as positive or negative, and determine which "
  "recurrent architecture (SimpleRNN, LSTM, GRU or BiLSTM) gives the best trade-off between "
  "classification performance, model size and training time.")
h2("1.4 Objectives")
bl(["To preprocess the IMDB movie review dataset into fixed-length integer sequences.",
    "To implement SimpleRNN, LSTM, GRU and BiLSTM classifiers with an identical overall architecture.",
    "To train all models under the same conditions and evaluate them on unseen test data.",
    "To compare the models using accuracy, precision, recall, F1-score, confusion matrices, training time and number of parameters.",
    "To test the trained models on new, user-written reviews and analyse their behaviour on difficult cases such as negation."])
h2("1.5 Scope and Purpose")
p("The project is limited to binary (positive or negative) sentiment classification of English movie "
  "reviews. It focuses on lightweight recurrent models that can be trained on a normal laptop CPU in a "
  "few minutes, rather than very large Transformer models. Its purpose is to give a clear, reproducible "
  "comparison that explains why gated recurrent units are preferred over the basic RNN in practice.")
h2("1.6 Related Work")
p("Maas et al. [1] introduced the IMDB Large Movie Review dataset and learned word vectors for sentiment "
  "analysis. Bengio et al. [6] showed that learning long-term dependencies with gradient descent is "
  "difficult because of vanishing and exploding gradients. Hochreiter and Schmidhuber [2] proposed the "
  "LSTM to solve this problem with a gated memory cell. Cho et al. [3] introduced the GRU in the RNN "
  "encoder-decoder framework, and Chung et al. [4] found that GRU and LSTM perform comparably on "
  "sequence modelling. Schuster and Paliwal [5] proposed bidirectional RNNs. More recently, Transformer "
  "models such as BERT [7] reach higher accuracy but require far more computation and memory.")

# --------------------------------------------------------------- 2
h1("2. Methodology")
h2("2.1 Approach")
p("The project follows an experimental, comparative approach. A single pipeline is built in which every "
  "step (data, preprocessing, embedding, classifier head, optimiser, number of epochs and random seed) is "
  "fixed, and only the recurrent layer is changed. Any difference in the results can therefore be "
  "attributed to the recurrent architecture itself. Figure 2 shows the complete workflow.")
h2("2.2 Dataset")
p("The IMDB Large Movie Review dataset [1] contains 50,000 highly polar movie reviews labelled positive "
  "(1) or negative (0). It is balanced, with 25,000 reviews of each class, so accuracy is a reliable "
  "metric. The standard split was used: 25,000 reviews for training, of which 20% (5,000) were held out "
  "as a validation set, and 25,000 reviews for testing. The test reviews are never seen during training. "
  "The dataset is loaded directly through the Keras datasets API, in which every word is already mapped "
  "to an integer index ranked by frequency. Figure 1 shows the distribution of review lengths: most "
  "reviews are shorter than 300 words, with a long tail of very long reviews.")
fig("fig0_review_lengths.png", "Distribution of review lengths in the training set (dashed line: chosen sequence length of 200)", 5.5)
h2("2.3 Data Preprocessing")
bl(["Vocabulary restriction: only the 10,000 most frequent words are kept, and rarer words are replaced by an 'unknown' token. This removes noise and keeps the embedding small.",
    "Integer encoding: each word is represented by its integer index in the vocabulary.",
    "Padding and truncation: neural networks need fixed-size inputs, so every review is padded with zeros or truncated to exactly 200 tokens. Pre-padding is used so that the last words of the review stay closest to the output."])
h2("2.4 Model Architectures")
p("All four models share the structure: Input (200 tokens) -> Embedding (10,000 x 64) -> Recurrent layer "
  "(64 units) -> Dropout (0.5) -> Dense (1 unit, sigmoid). The embedding layer learns a 64-dimensional "
  "vector for each word, in which words with similar meaning get similar vectors.")
p("SimpleRNN. At each time step t the hidden state is updated from the current input x_t and the previous hidden state:")
eq(r"h_t = \tanh(W_x x_t + W_h h_{t-1} + b)", "h_t = tanh(W_x·x_t + W_h·h_(t-1) + b)")
p("When the network is unfolded over 200 time steps, the gradient is multiplied by W_h many times during "
  "backpropagation through time, so it vanishes and early words receive almost no learning signal.")
p("LSTM. The LSTM adds a separate cell state c_t that information can flow along almost unchanged, "
  "controlled by three gates (forget f_t, input i_t and output o_t), where sigma is the sigmoid function:")
eq(r"f_t = \sigma(W_f[h_{t-1},x_t]+b_f),\quad i_t = \sigma(W_i[h_{t-1},x_t]+b_i),\quad o_t = \sigma(W_o[h_{t-1},x_t]+b_o)",
   "f_t = σ(W_f[h_(t-1), x_t] + b_f),   i_t = σ(W_i[h_(t-1), x_t] + b_i),   o_t = σ(W_o[h_(t-1), x_t] + b_o)")
eq(r"c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c[h_{t-1},x_t]+b_c),\qquad h_t = o_t \odot \tanh(c_t)",
   "c_t = f_t ⊙ c_(t-1) + i_t ⊙ tanh(W_c[h_(t-1), x_t] + b_c),     h_t = o_t ⊙ tanh(c_t)")
p("GRU. The GRU merges the cell state and hidden state and uses only two gates, update z_t and reset r_t, "
  "so it has fewer parameters than the LSTM:")
eq(r"z_t = \sigma(W_z[h_{t-1},x_t]),\quad r_t = \sigma(W_r[h_{t-1},x_t]),\quad h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tanh(W[r_t \odot h_{t-1}, x_t])",
   "z_t = σ(W_z[h_(t-1), x_t]),  r_t = σ(W_r[h_(t-1), x_t]),  h_t = (1 − z_t) ⊙ h_(t-1) + z_t ⊙ tanh(W[r_t ⊙ h_(t-1), x_t])")
p("BiLSTM. Two independent LSTMs read the review forwards and backwards, and their final hidden states are "
  "concatenated, so every position has context from both sides. The recurrent layer here acts like the "
  "encoder of an encoder-decoder (Seq2Seq) model: it compresses the whole review into one fixed-length "
  "vector, which is then decoded by the dense layer into a single label (a many-to-one architecture).")
p("Output layer. The final dense layer outputs y = sigmoid(w·h + b), the probability that the review is "
  "positive. A threshold of 0.5 gives the predicted class.")
h2("2.5 Experimental Setup")
tab(["Hyperparameter", "Value"],
    [["Vocabulary size", "10,000"], ["Sequence length", "200 tokens"], ["Embedding dimension", "64"],
     ["Recurrent units", "64 (64 per direction for BiLSTM)"], ["Dropout", "0.5"],
     ["Optimizer", "Adam (learning rate 0.001)"], ["Loss function", "Binary cross-entropy"],
     ["Batch size", "128"], ["Maximum epochs", "5"],
     ["Early stopping", "Validation loss, patience 2, best weights restored"], ["Random seed", "42"]],
    "Hyperparameters used for all four models")
p("Justification of choices: Adam adapts the learning rate of every parameter and converges quickly "
  "without tuning [8]. Binary cross-entropy is the standard loss for a sigmoid output with two classes. "
  "Dropout [9] and early stopping are used because recurrent models overfit IMDB within a few epochs. "
  "64 units and 5 epochs keep the training time of each model within a few minutes on a CPU.")
eq(r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]",
   "Loss = −(1/N) Σ [ y_i·log(ŷ_i) + (1 − y_i)·log(1 − ŷ_i) ]")
h2("2.6 Evaluation Metrics")
bl(["Accuracy = (TP + TN) / (TP + TN + FP + FN): the fraction of reviews classified correctly.",
    "Precision = TP / (TP + FP): of the reviews predicted positive, how many really are positive.",
    "Recall = TP / (TP + FN): of the truly positive reviews, how many were found.",
    "F1-score = 2 x Precision x Recall / (Precision + Recall): the harmonic mean of precision and recall.",
    "Confusion matrix, training time (seconds) and number of trainable parameters."])
h2("2.7 Software Requirements Specifications")
tab(["Component", "Specification"],
    [["Operating system", "Windows 11 (64-bit)"], ["Programming language", "Python 3.11"],
     ["Deep learning framework", "TensorFlow 2 / Keras"], ["Data handling", "NumPy, Pandas"],
     ["Evaluation", "scikit-learn"], ["Visualisation", "Matplotlib, Seaborn"],
     ["Hardware", "Standard laptop, CPU only (no GPU required), 8 GB RAM or more"],
     ["Dataset", "IMDB Large Movie Review dataset (via keras.datasets)"]],
    "Software and hardware requirements")

# --------------------------------------------------------------- 3
h1("3. Implementation")
h2("3.1 Implementation Steps")
bl(["Step 1 - Environment setup: installed Python 3.11 with TensorFlow, scikit-learn, Pandas, Matplotlib and Seaborn, and fixed all random seeds to 42 for reproducibility.",
    "Step 2 - Data loading: loaded the IMDB dataset with a 10,000-word vocabulary using keras.datasets.imdb.",
    "Step 3 - Preprocessing: padded or truncated every review to 200 tokens with pad_sequences.",
    "Step 4 - Model construction: wrote a single build(kind) function that returns a SimpleRNN, LSTM, GRU or BiLSTM model with the same embedding, dropout and output layers.",
    "Step 5 - Training: trained each model with Adam, binary cross-entropy, batch size 128, a 20% validation split and early stopping, recording the training time.",
    "Step 6 - Evaluation: predicted the 25,000 test reviews and computed accuracy, precision, recall, F1-score and the confusion matrix.",
    "Step 7 - Visualisation: generated the training curves, metric comparison chart, confusion matrices and time/parameter charts.",
    "Step 8 - Prediction on new reviews: encoded user-written sentences with the IMDB word index and classified them with all four models."])
h2("3.2 Design (DFD Diagram)")
p("Figure 2 shows the data flow of the system, from the raw dataset to the final comparison.")
fig("fig5_system_design_dfd.png", "System design / data flow diagram of the proposed comparative framework", 4.6)
h2("3.3 Code Snippets")
p("The core of the implementation is one model-building function, so all four models are guaranteed to be identical except for the recurrent layer:")
code("""def build(kind):
    m = models.Sequential(name=kind)
    m.add(layers.Input(shape=(MAX_LEN,)))
    m.add(layers.Embedding(VOCAB_SIZE, EMBED_DIM))
    if kind == "SimpleRNN": m.add(layers.SimpleRNN(UNITS))
    elif kind == "LSTM":    m.add(layers.LSTM(UNITS))
    elif kind == "GRU":     m.add(layers.GRU(UNITS))
    elif kind == "BiLSTM":  m.add(layers.Bidirectional(layers.LSTM(UNITS)))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])
    return m""")
p("Each model is trained with early stopping and then evaluated on the test set:")
code("""es = callbacks.EarlyStopping(monitor="val_loss", patience=2,
                             restore_best_weights=True)
h = model.fit(x_train, y_train, epochs=5, batch_size=128,
              validation_split=0.2, callbacks=[es])
pred = (model.predict(x_test) >= 0.5).astype(int)
print(accuracy_score(y_test, pred), f1_score(y_test, pred))""")
h2("3.4 Challenges and Solutions")
bl(["Variable review length: reviews range from about 10 to more than 2,000 words. Solved by padding or truncating to 200 tokens, which covers the majority of reviews (Figure 1).",
    "Overfitting: training accuracy rose to about 93% while validation loss started increasing after epoch 2. Solved with dropout (0.5) and early stopping that restores the best weights.",
    "Long training time on CPU: BiLSTM takes around 130 seconds per epoch. Solved by using 64 units, a batch size of 128 and at most 5 epochs.",
    "Fair comparison: different random initialisations could change the ranking. Solved by fixing the random seed and using the same build function and settings for every model.",
    "Encoding new sentences: the IMDB dataset uses an index offset of 3 for special tokens. Solved by shifting the Keras word index by 3 and mapping out-of-vocabulary words to the 'unknown' token."])

# --------------------------------------------------------------- 4
h1("4. Result and Analysis")
h2("4.1 Overall Performance")
p("Table 3 shows the performance of the four models on the 25,000 unseen test reviews, and Figure 3 compares the metrics visually.")
rows = [[r["Model"], f"{int(r['Parameters']):,}", f"{r['Test Accuracy']:.4f}", f"{r['Precision']:.4f}",
         f"{r['Recall']:.4f}", f"{r['F1-score']:.4f}", f"{r['Train time (s)']:.1f}"] for _, r in df.iterrows()]
tab(["Model", "Parameters", "Accuracy", "Precision", "Recall", "F1-score", "Train time (s)"], rows,
    "Performance of the four models on the IMDB test set (25,000 reviews)")
fig("fig2_metric_comparison.png", "Test-set accuracy, precision, recall and F1-score of each model")
p(f"LSTM achieves the best accuracy ({acc('LSTM')}) and precision ({g['LSTM']['Precision']:.3f}). BiLSTM "
  f"achieves the best recall ({g['BiLSTM']['Recall']:.3f}) and F1-score ({g['BiLSTM']['F1-score']:.3f}). "
  f"GRU follows closely with {acc('GRU')}, and SimpleRNN is clearly the weakest at {acc('SimpleRNN')}.")
h2("4.2 Training Behaviour")
fig("fig1_training_curves.png", "Training (dashed) and validation (solid) accuracy and loss per epoch")
p("Figure 4 shows that all models reach their lowest validation loss at epoch 2. After that, training "
  "accuracy keeps rising towards 93% while validation loss increases, which is the typical sign of "
  "overfitting. Early stopping detected this and restored the epoch-2 weights for every model. SimpleRNN "
  "learns slowest (65% training accuracy after epoch 1) and has the highest validation loss throughout, "
  "which reflects its difficulty in using long-range context.")
h2("4.3 Confusion Matrices")
fig("fig3_confusion_matrices.png", "Confusion matrices of the four models on the test set")
p("The confusion matrices (Figure 5) show that the errors are fairly balanced between the two classes, "
  "so none of the models is biased towards one class. LSTM makes the fewest false positives (1,466), "
  "while BiLSTM makes the fewest false negatives (1,556), which explains its higher recall.")
h2("4.4 Computational Cost")
fig("fig4_time_and_params.png", "Training time and number of trainable parameters")
p(f"SimpleRNN is the fastest ({g['SimpleRNN']['Train time (s)']:.0f} s) because it has a single weight "
  f"matrix per step. LSTM ({g['LSTM']['Train time (s)']:.0f} s) and GRU ({g['GRU']['Train time (s)']:.0f} s) "
  f"are about 3 times slower, and GRU is about 8% faster than LSTM. BiLSTM is the most expensive "
  f"({g['BiLSTM']['Train time (s)']:.0f} s), about 1.8 times LSTM, because it runs two LSTMs. Most "
  f"parameters belong to the embedding layer (640,000), so the recurrent layers differ only by 25,000 to "
  f"58,000 parameters.")
h2("4.5 Predictions on New Reviews")
dr = [[r["Review"]] + [r[m] for m in ["SimpleRNN", "LSTM", "GRU", "BiLSTM"]] for _, r in demo.iterrows()]
tab(["Review (written by us)", "SimpleRNN", "LSTM", "GRU", "BiLSTM"], dr,
    "Predicted label and probability of positive sentiment for new reviews")
p("All models classify the clearly positive and clearly negative reviews correctly. On the mixed review "
  "(\"the plot was predictable but the music and visuals were great\"), only the gated models understand "
  "that the clause after \"but\" dominates, while SimpleRNN predicts negative. The negated sentence "
  "(\"it was not good\") is the hardest case: GRU and BiLSTM classify it correctly, but LSTM gives a "
  "borderline 0.51. This shows that negation is still a challenge for small recurrent models.")

# --------------------------------------------------------------- 5
h1("5. Discussion and Conclusion")
h2("5.1 Comparison with Objectives")
p("All objectives were achieved: the IMDB data was preprocessed, four recurrent models were implemented "
  "with an identical pipeline, trained under the same conditions, evaluated with six measures and tested "
  "on new reviews. The initial hypothesis, that gated architectures outperform the basic RNN on long "
  "text, was confirmed.")
h2("5.2 Implications of the Findings")
bl([f"Gating matters: LSTM, GRU and BiLSTM beat SimpleRNN by 2.4 to 3.6 percentage points, because their gates prevent the vanishing gradient problem and keep sentiment cues from early in the review.",
    f"GRU is the most efficient choice: it reaches {acc('GRU')}, only 1.2 points below LSTM, with fewer parameters and less training time, which makes it suitable for mobile or low-resource deployment.",
    "Bidirectionality is not worth its cost here: BiLSTM gives the best F1-score but no accuracy gain over LSTM, for 1.8 times the training time. For whole-document classification, the final forward state already summarises the review well.",
    "Regularisation is essential: every model starts to overfit after two epochs, so early stopping and dropout are necessary even for small models."])
h2("5.3 Limitations and Sources of Error")
bl(["Each model was trained once with a single random seed. Small differences such as LSTM versus BiLSTM (0.14 points) may fall within run-to-run variance.",
    "Reviews were truncated to 200 tokens, so information at the start of very long reviews is lost.",
    "Embeddings were learned from scratch. Pretrained embeddings could improve all models.",
    "Only binary sentiment was studied. Neutral, mixed and sarcastic opinions are not modelled.",
    "The models struggle with negation, as shown in the new-review test."])
h2("5.4 Conclusion")
p(f"This project compared SimpleRNN, LSTM, GRU and BiLSTM for sentiment analysis of IMDB movie reviews "
  f"under identical conditions. LSTM achieved the best accuracy ({acc('LSTM')}), BiLSTM the best F1-score, "
  "and GRU nearly matched them at lower cost, while the basic RNN performed worst. The study shows "
  "experimentally why gated recurrent units replaced the basic RNN in practice, and that GRU gives the "
  "best accuracy-to-cost ratio. Its contribution is a simple, fully reproducible benchmark and code base "
  "that can serve as a baseline for more advanced models.")

# --------------------------------------------------------------- 6
h1("6. Future Scope")
bl(["Pretrained word embeddings: initialise the embedding layer with GloVe or Word2Vec vectors to improve accuracy and generalisation.",
    "Attention mechanism: add attention on top of the LSTM/GRU so the model can focus on the most important words, which also makes predictions explainable.",
    "Transformer models: compare with BERT, DistilBERT or RoBERTa to quantify the accuracy-versus-cost gap.",
    "Stacked and hybrid models: deeper recurrent stacks and CNN-LSTM hybrids.",
    "Multi-class and aspect-based sentiment: predict neutral or star ratings, and opinions about specific aspects such as acting or music.",
    "Multilingual sentiment: extend to Indian languages such as Telugu and Hindi, and to code-mixed social media text.",
    "Statistical robustness: repeat the experiments over several random seeds and report the mean and standard deviation with significance tests.",
    "Deployment: package the GRU model as a web or mobile application for real-time review analysis."])

REFS = [
    "A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng and C. Potts, \"Learning Word Vectors for Sentiment Analysis,\" in Proc. 49th Annual Meeting of the Association for Computational Linguistics (ACL), 2011, pp. 142-150.",
    "S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997. doi: 10.1162/neco.1997.9.8.1735",
    "K. Cho et al., \"Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation,\" in Proc. EMNLP, 2014, pp. 1724-1734.",
    "J. Chung, C. Gulcehre, K. Cho and Y. Bengio, \"Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling,\" arXiv:1412.3555, 2014.",
    "M. Schuster and K. K. Paliwal, \"Bidirectional Recurrent Neural Networks,\" IEEE Transactions on Signal Processing, vol. 45, no. 11, pp. 2673-2681, 1997.",
    "Y. Bengio, P. Simard and P. Frasconi, \"Learning Long-Term Dependencies with Gradient Descent is Difficult,\" IEEE Transactions on Neural Networks, vol. 5, no. 2, pp. 157-166, 1994.",
    "J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, \"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,\" in Proc. NAACL-HLT, 2019, pp. 4171-4186.",
    "D. P. Kingma and J. Ba, \"Adam: A Method for Stochastic Optimization,\" in Proc. International Conference on Learning Representations (ICLR), 2015.",
    "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, \"Dropout: A Simple Way to Prevent Neural Networks from Overfitting,\" Journal of Machine Learning Research, vol. 15, pp. 1929-1958, 2014.",
    "F. Chollet et al., \"Keras,\" 2015. [Online]. Available: https://keras.io",
    "I. Goodfellow, Y. Bengio and A. Courville, Deep Learning. MIT Press, 2016, ch. 10.",
]
APPENDIX = [
    ("A.1 Project Files", ["sentiment_rnn_lstm_gru.py - complete training and evaluation code",
                           "results/results_table.csv - final metrics of all models",
                           "results/demo_predictions.csv - predictions on new reviews",
                           "results/fig0 to fig5 - all figures used in this report"]),
    ("A.2 How to Run", ["pip install tensorflow scikit-learn pandas matplotlib seaborn",
                        "python sentiment_rnn_lstm_gru.py  (about 20 minutes on a laptop CPU)"]),
]

# =====================================================================  DOCX
FONT = "Times New Roman"

def set_text(par, text):
    """Replace a paragraph's text but keep the first run's formatting."""
    runs = par.runs
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""

def build_docx():
    d = docx.Document(TEMPLATE)
    body = d.element.body
    els = list(body.iterchildren())
    P = lambda i: docx.text.paragraph.Paragraph(els[i], d)

    # --- cover page
    set_text(P(0), TITLE)
    P(11).runs[0].text = f"{STUDENTS[0][0]}  {STUDENTS[0][1]}"
    P(12).runs[0].text = f"{STUDENTS[1][0]}  {STUDENTS[1][1]}"
    P(21).runs[0].text = MONTH
    # --- certificate
    c = P(26).runs
    c[1].text = TITLE
    c[3].text = f"{STUDENTS[0][0].title()} ({STUDENTS[0][1]}) and {STUDENTS[1][0].title()} ({STUDENTS[1][1]})"
    c[4].text = c[4].text.replace("Bachelor of Technology/Master of Technology", "Bachelor of Technology")
    P(34).runs[0].text = GUIDE
    P(35).runs[0].text = GUIDE_DESIG + ","
    P(36).runs[0].text = "SRM University – AP."
    # --- acknowledgement
    a = P(41).runs
    a[0].text = a[0].text.replace("I am extremely", "We are extremely").replace("my project", "our project").replace("express my", "express our")
    a[1].text = GUIDE
    a[2].text = (a[2].text.replace("for her kind help", "for the kind help")
                 .replace("giving me", "giving us").replace("Andhra pradesh", "Andhra Pradesh"))
    P(42).runs[0].text = "\n" + f"{STUDENTS[0][0].title()}  ({STUDENTS[0][1]})"
    P(43).runs[0].text = f"{STUDENTS[1][0].title()}  ({STUDENTS[1][1]})\n\n"

    # --- remove the template's sample body (keep everything up to the TOC page break)
    for el in els[59:]:
        if el.tag != qn("w:sectPr"):
            body.remove(el)

    fig_no, tab_no, first = [0], [0], [True]
    fig_order = {}

    def para(text="", bold=False, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, style=None, italic=False, font=FONT):
        q = d.add_paragraph(style=style)
        q.alignment = align
        q.paragraph_format.space_after = Pt(6)
        if style is None:
            q.paragraph_format.line_spacing = 1.5
        if text:
            r = q.add_run(text); r.bold = bold; r.italic = italic
            r.font.size = Pt(size); r.font.name = font
            r._element.rPr.rFonts.set(qn("w:eastAsia"), font)
        return q

    def page_break():
        q = d.add_paragraph(); q.add_run().add_break(WD_BREAK.PAGE)

    for blk in C:
        k = blk[0]
        if k == "h1":
            if not first[0]:
                page_break()
            first[0] = False
            q = d.add_paragraph(style="Heading 1"); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = q.add_run(blk[1]); r.font.name = FONT; r.font.size = Pt(18)
        elif k == "h2":
            q = d.add_paragraph(style="Heading 2")
            q.paragraph_format.space_before = Pt(12)
            r = q.add_run(blk[1]); r.font.name = FONT; r.font.size = Pt(14)
        elif k == "p":
            if blk[1].startswith("Keywords:"):
                q = para(); r = q.add_run("Keywords: "); r.bold = True; r.font.name = FONT
                r2 = q.add_run(blk[1][10:]); r2.italic = True; r2.font.name = FONT
            else:
                para(blk[1])
        elif k == "bullets":
            for it in blk[1]:
                q = para()
                q.paragraph_format.left_indent = Inches(0.35)
                q.paragraph_format.first_line_indent = Inches(-0.25)
                q.paragraph_format.space_after = Pt(3)
                r = q.add_run("•  "); r.font.name = FONT; r.font.size = Pt(12)
                if " - " in it[:40] or ": " in it[:45]:
                    sep = " - " if " - " in it[:40] else ": "
                    head, rest = it.split(sep, 1)
                    r = q.add_run(head + sep); r.bold = True; r.font.name = FONT; r.font.size = Pt(12)
                    r = q.add_run(rest); r.font.name = FONT; r.font.size = Pt(12)
                else:
                    r = q.add_run(it); r.font.name = FONT; r.font.size = Pt(12)
        elif k == "eq":
            para(blk[2], italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=11, font="Cambria Math")
        elif k == "code":
            q = d.add_paragraph(); q.paragraph_format.left_indent = Inches(0.3)
            r = q.add_run(blk[1]); r.font.name = "Consolas"; r.font.size = Pt(9)
            shd = docx.oxml.OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "F2F2F2")
            q._p.get_or_add_pPr().append(shd)
        elif k == "fig":
            fig_no[0] += 1
            d.add_picture(f"{R}/{blk[1]}", width=Inches(blk[3]))
            d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            para(f"Figure {fig_no[0]}. {blk[2]}", bold=True, size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
        elif k == "table":
            tab_no[0] += 1
            para(f"Table {tab_no[0]}. {blk[3]}", bold=True, size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
            head, rows = blk[1], blk[2]
            t = d.add_table(rows=1, cols=len(head))
            t.alignment = 1
            tblPr = t._tbl.tblPr
            borders = docx.oxml.OxmlElement("w:tblBorders")
            for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
                b = docx.oxml.OxmlElement(f"w:{side}")
                b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "6"); b.set(qn("w:color"), "808080")
                borders.append(b)
            tblPr.append(borders)
            for i, h in enumerate(head):
                cell = t.rows[0].cells[i]; cell.text = ""
                r = cell.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(10); r.font.name = FONT
                shd = docx.oxml.OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "D9E2F3")
                cell._tc.get_or_add_tcPr().append(shd)
            for row in rows:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = ""
                    r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(10); r.font.name = FONT
            para()

    # references
    page_break()
    q = d.add_paragraph(style="Heading 1"); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = q.add_run("References"); r.font.name = FONT; r.font.size = Pt(18)
    for i, ref in enumerate(REFS, 1):
        q = para(f"[{i}] {ref}", size=12)
        q.paragraph_format.left_indent = Inches(0.4); q.paragraph_format.first_line_indent = Inches(-0.4)
    # appendix
    page_break()
    q = d.add_paragraph(style="Heading 1"); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = q.add_run("Appendices"); r.font.name = FONT; r.font.size = Pt(18)
    for head, items in APPENDIX:
        q = d.add_paragraph(style="Heading 2"); r = q.add_run(head); r.font.name = FONT; r.font.size = Pt(14)
        for it in items:
            q = para(it, size=11, align=WD_ALIGN_PARAGRAPH.LEFT, font="Consolas" if "pip" in it or "python" in it else FONT)

    d.save(OUT_DOCX)
    print("Saved", OUT_DOCX)

# =====================================================================  LATEX
def esc(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    s = "".join(rep.get(ch, ch) for ch in s)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    s = s.replace("->", r"$\rightarrow$").replace(" x ", r" $\times$ ").replace("–", "--")
    return s

def build_tex():
    os.makedirs(f"{TEX_DIR}/figures", exist_ok=True)
    for f in os.listdir(R):
        if f.endswith(".png"):
            shutil.copy(f"{R}/{f}", f"{TEX_DIR}/figures/{f}")
    # extract the SRM logo from the template
    with zipfile.ZipFile(TEMPLATE) as z:
        d = docx.Document(TEMPLATE)
        el = list(d.element.body.iterchildren())[14]
        rid = el.findall(".//" + qn("a:blip"))[0].get(qn("r:embed"))
        part = d.part.related_parts[rid]
        ext = os.path.splitext(part.partname)[1]
        with open(f"{TEX_DIR}/figures/srm_logo{ext}", "wb") as fh:
            fh.write(part.blob)
    logo = f"srm_logo{ext}"

    L = []
    w = L.append
    w(r"""\documentclass[12pt,a4paper,oneside]{report}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{graphicx,float,booktabs,array,tabularx,setspace,amsmath}
\usepackage{enumitem,listings,xcolor,titlesec,caption}
\usepackage[hidelinks]{hyperref}
\graphicspath{{figures/}}
\onehalfspacing
\setlength{\parskip}{6pt}
\setlist{itemsep=2pt}
\captionsetup{font={bf,small},labelsep=period,figurename=Figure}
\titleformat{\chapter}[block]{\centering\bfseries\fontsize{18}{22}\selectfont}{}{0pt}{}
\titlespacing*{\chapter}{0pt}{0pt}{18pt}
\titleformat{\section}{\bfseries\fontsize{14}{17}\selectfont}{}{0pt}{}
\renewcommand{\thefigure}{\arabic{figure}}
\renewcommand{\thetable}{\arabic{table}}
\lstset{basicstyle=\ttfamily\footnotesize,backgroundcolor=\color{gray!10},frame=single,
  rulecolor=\color{gray!40},keywordstyle=\color{blue!60!black},language=Python,
  showstringspaces=false,xleftmargin=1em}
\begin{document}
""")
    # --- cover page (mirrors the Word template)
    w(r"""\begin{titlepage}
\centering
{\fontsize{16}{20}\selectfont\bfseries %s\par}
\vspace{1.2cm}
Project submitted to the\\
SRM University -- AP, Andhra Pradesh\\
for the partial fulfillment of the requirements to award the degree of\\[0.6cm]
{\bfseries Bachelor of Technology}\\
In\\
{\bfseries Computer Science and Engineering}\\
{\bfseries School of Engineering and Sciences}\\[1cm]
{\bfseries %s \quad %s}\\
{\bfseries %s \quad %s}\\[1cm]
\includegraphics[width=0.38\textwidth]{%s}\\[0.8cm]
Under the Guidance of\\
{\bfseries %s}\\
%s\\
SRM University--AP\\
Neerukonda, Mangalagiri, Guntur\\
Andhra Pradesh -- 522 240\\[0.8cm]
{\bfseries %s}
\end{titlepage}
""" % (esc(TITLE), STUDENTS[0][0], STUDENTS[0][1], STUDENTS[1][0], STUDENTS[1][1], logo,
       GUIDE, GUIDE_DESIG, MONTH))
    # --- certificate
    w(r"""\begin{center}{\fontsize{19}{23}\selectfont\bfseries Certificate}\end{center}
\vspace{0.5cm}
\hfill Date: \underline{\hspace{3cm}}
\vspace{1cm}

This is to certify that the work present in this Project entitled ``\textbf{%s}'' has been carried out by \textbf{%s (%s) and %s (%s)} under my supervision. The work is genuine, original, and suitable for submission to the SRM University -- AP for the award of Bachelor of Technology in \textbf{School of Engineering and Sciences}.

\vspace{3cm}
\noindent\textbf{Supervisor}\\[1cm]
(Signature)\\
%s\\
%s,\\
SRM University -- AP.
\newpage
""" % (esc(TITLE), STUDENTS[0][0].title(), STUDENTS[0][1], STUDENTS[1][0].title(), STUDENTS[1][1],
       GUIDE, GUIDE_DESIG))
    # --- acknowledgement
    w(r"""\begin{center}{\fontsize{18}{22}\selectfont\bfseries Acknowledgement}\end{center}
\vspace{0.5cm}

The satisfaction that accompanies the successful completion of any task would be incomplete without introducing the people who made it possible and whose constant guidance and encouragement crowns all efforts with success.

We are extremely grateful and express our profound gratitude and indebtedness to our project guide, \textbf{%s}, Department of Computer Science \& Engineering, SRM University, Andhra Pradesh, for the kind help and for giving us the necessary guidance and valuable suggestions in completing this project work.

\vspace{2cm}
\begin{center}
%s (%s)\\
%s (%s)
\end{center}
\newpage
\tableofcontents
\newpage
""" % (GUIDE, STUDENTS[0][0].title(), STUDENTS[0][1], STUDENTS[1][0].title(), STUDENTS[1][1]))

    in_list = False
    for blk in C:
        k = blk[0]
        if k == "h1":
            w(r"\chapter*{%s}" % esc(blk[1]))
            w(r"\addcontentsline{toc}{chapter}{%s}" % esc(blk[1]))
        elif k == "h2":
            w(r"\section*{%s}" % esc(blk[1]))
            w(r"\addcontentsline{toc}{section}{%s}" % esc(blk[1]))
        elif k == "p":
            if blk[1].startswith("Keywords:"):
                w(r"\noindent\textbf{Keywords:} \textit{%s}" % esc(blk[1][10:]) + "\n")
            else:
                w(esc(blk[1]) + "\n")
        elif k == "bullets":
            w(r"\begin{itemize}")
            for it in blk[1]:
                sep = " - " if " - " in it[:40] else (": " if ": " in it[:45] else None)
                if sep:
                    head, rest = it.split(sep, 1)
                    w(r"\item \textbf{%s%s}%s" % (esc(head), esc(sep.strip() if sep == ": " else " --"), " " + esc(rest)))
                else:
                    w(r"\item " + esc(it))
            w(r"\end{itemize}")
        elif k == "eq":
            w(r"\begin{equation*}" + blk[1] + r"\end{equation*}")
        elif k == "code":
            w(r"\begin{lstlisting}" + "\n" + blk[1] + "\n" + r"\end{lstlisting}")
        elif k == "fig":
            frac = min(blk[3] / 6.27, 1.0)
            w(r"""\begin{figure}[H]\centering
\includegraphics[width=%.2f\textwidth]{%s}
\caption{%s}
\end{figure}""" % (frac, blk[1], esc(blk[2])))
        elif k == "table":
            head, rows = blk[1], blk[2]
            n = len(head)
            if n == 5:   # demo table with long first column
                spec = r">{\raggedright\arraybackslash}X" + "c" * 4
            elif n == 2:
                spec = r"l>{\raggedright\arraybackslash}X"
            else:
                spec = "l" + "c" * (n - 1)
            env = "tabularx" if n in (2, 5) else "tabular"
            width = r"{\textwidth}" if env == "tabularx" else ""
            w(r"\begin{table}[H]\centering\small")
            w(r"\caption{%s}" % esc(blk[3]))
            w(r"\begin{%s}%s{%s}\toprule" % (env, width, spec))
            w(" & ".join(r"\textbf{%s}" % esc(h) for h in head) + r" \\\midrule")
            for row in rows:
                w(" & ".join(esc(str(v)) for v in row) + r" \\")
            w(r"\bottomrule\end{%s}\end{table}" % env)

    w(r"\chapter*{References}\addcontentsline{toc}{chapter}{References}")
    w(r"\begin{enumerate}[label={[\arabic*]},leftmargin=2.2em]")
    for ref in REFS:
        ref = esc(ref).replace("https://keras.io", r"\url{https://keras.io}")
        w(r"\item " + ref)
    w(r"\end{enumerate}")
    w(r"\chapter*{Appendices}\addcontentsline{toc}{chapter}{Appendices}")
    for head, items in APPENDIX:
        w(r"\section*{%s}" % esc(head))
        w(r"\begin{itemize}")
        for it in items:
            w(r"\item \texttt{%s}" % esc(it) if ("pip" in it or "python" in it) else r"\item " + esc(it))
        w(r"\end{itemize}")
    w(r"\end{document}")
    with open(f"{TEX_DIR}/report.tex", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("Saved", f"{TEX_DIR}/report.tex")

if __name__ == "__main__":
    build_docx()
    build_tex()
