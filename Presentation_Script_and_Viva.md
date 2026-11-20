# Presentation Script & Viva Guide

**Project:** A Comparative Study of RNN, LSTM, GRU and BiLSTM Models for Sentiment Analysis of IMDB Movie Reviews
**Team:** Harsha Siva Akash Saladi (AP23110011490), Kishan Bhimavarapu (AP23110011351)
**Guide:** Dr. Lalit Kumar, Assistant Professor, Department of CSE, SRM University – AP

> Read the companion file **[Concepts_and_Formulas.md](Concepts_and_Formulas.md)** first. It explains every concept and formula used here from zero.

---

## PART 0 – What to Run and Where (do this before the presentation)

### Where to run
Everything runs in **PowerShell** (or the VS Code terminal) **inside the project folder**:
```
D:\Hirishi Btech\My Projects\DEEP LEARNING PROJECT
```
To open it: in File Explorer, open the folder, click the address bar, type `powershell` and press Enter. Or in VS Code use **File → Open Folder**, then **Terminal → New Terminal**.

### Step 1 – Install the libraries (only once, on a new PC)
```powershell
pip install -r requirements.txt
```

### Step 2 – Train all 4 models and make all the results (about 20 minutes)
```powershell
python sentiment_rnn_lstm_gru.py
```
**What you'll see:** each model training epoch by epoch, then a final results table and the predictions on new reviews.
**What it creates:** the `results/` folder:

| File | What it shows | When to show it |
|---|---|---|
| `results/fig0_review_lengths.png` | Review length histogram | Slide 4 – Dataset |
| `results/fig5_system_design_dfd.png` | System design / DFD | Slide 5 – Design |
| `results/fig2_metric_comparison.png` | Accuracy/Precision/Recall/F1 bar chart | Slide 7 – Results |
| `results/fig1_training_curves.png` | Training vs validation curves (overfitting) | Slide 7 – Results |
| `results/fig3_confusion_matrices.png` | Confusion matrices | Slide 7 – Results |
| `results/fig4_time_and_params.png` | Training time and parameters | Slide 7 – Results |
| `results/results_table.csv` | Final numbers table | Slide 7 – Results |
| `results/demo_predictions.csv` | 4 models on our own sentences | Slide 8 – Demo |

> ⚠️ You do **not** need to run Step 2 again during the presentation, because the results are already saved. Just open the images from the `results/` folder. Run it only if the examiner asks you to show the training.

### Step 3 – Live demo: type your own review (the examiner's favourite)
```powershell
python predict.py
```
Then type any review and press Enter, for example:
```
Review: The story was boring and the acting was terrible
  -> NEGATIVE 😠   (probability of positive = 0.04)
Review: q
```
Or give one sentence directly:
```powershell
python predict.py "This movie was absolutely wonderful"
```
The first run trains and saves the GRU model (about 2–3 minutes). After that it loads instantly from `models/gru_demo.keras`. **Run it once before the presentation.**

### Step 4 – Rebuild the reports (only if you change something)
```powershell
python make_dfd.py
python build_reports.py
```
This creates `Final_Project_Report.docx` and `Report_LaTeX/report.tex`. After rebuilding, open the Word file and right-click the Table of Contents → **Update Field**.

### Which file to submit or open

| What | File |
|---|---|
| College project report (Word) | `Final_Project_Report.docx` |
| College project report (PDF) | `Final_Project_Report.pdf` |
| Same report in LaTeX | `Report_LaTeX/report.tex` (upload the `Report_LaTeX` folder to Overleaf) |
| IEEE research paper | `IEEE_Paper/main.tex` and `IEEE_Paper/main.pdf` |
| Main code to show | `sentiment_rnn_lstm_gru.py` |

### Order on presentation day
1. Open PowerShell in the project folder and run `python predict.py` once, so it's ready.
2. Open `Final_Project_Report.pdf`. Its figures are your slides.
3. Follow the script in Part 1 below.
4. At Slide 8, switch to PowerShell and type 2–3 reviews live.

---

## Final results (memorise this table)

| Model | Accuracy | Precision | Recall | F1 | Train time | Parameters |
|---|---|---|---|---|---|---|
| SimpleRNN | 83.12% | 0.829 | 0.834 | 0.832 | 95 s | 648,321 |
| **LSTM** | **86.76%** ⭐ | **0.879** | 0.853 | 0.866 | 296 s | 673,089 |
| GRU | 85.54% | 0.857 | 0.854 | 0.855 | 272 s | 665,025 |
| BiLSTM | 86.62% | 0.860 | **0.876** | **0.867** ⭐ | 537 s | 706,177 |

**Three key findings:**
1. Gated models (LSTM, GRU, BiLSTM) beat the basic RNN by **2.4–3.6%**, because gates fix the vanishing gradient problem.
2. **GRU ≈ LSTM**: only 1.2% lower accuracy, but fewer parameters and 8% faster. It's the best value.
3. **BiLSTM is not worth it here**: same accuracy as LSTM for **1.8× the training time**.

---

## PART 1 – Presentation Script (about 7 minutes)

### Slide 1 – Title (20 sec)
"Good morning, respected sir and everyone. We are Harsha Siva Akash Saladi and Kishan Bhimavarapu. Our project is **'A Comparative Study of RNN, LSTM, GRU and BiLSTM Models for Sentiment Analysis of IMDB Movie Reviews'**, under the guidance of Dr. Lalit Kumar."

### Slide 2 – What is sentiment analysis? (40 sec)
"Sentiment analysis means teaching a computer to read a text and decide whether the opinion in it is **positive or negative**.
For example, *'This movie was wonderful'* is positive and *'Worst film ever'* is negative.
It's used by Amazon and Flipkart for product reviews, by companies to track social media, and by IMDB for movie ratings."

### Slide 3 – Problem statement (40 sec)
"The difficulty is that **word order changes meaning**. *'Good'* is positive, but *'not good'* is negative.
Normal neural networks look at all the words at once and ignore order. **Recurrent Neural Networks** read one word at a time and remember the previous words.
But there are four main types of recurrent network, so our question is: **which one works best, and is the extra training time worth it?**"

### Slide 4 – Dataset (40 sec)
*(Show Figure 1, the review-length histogram)*
"We used the **IMDB dataset**: 50,000 real movie reviews, 25,000 positive and 25,000 negative.
- 20,000 for training, 5,000 for validation and 25,000 for testing.
- We kept the 10,000 most common words, and made every review exactly 200 words long by padding or cutting, because a neural network needs inputs of the same size. The histogram shows most reviews are under 300 words."

### Slide 5 – System design (1 min)
*(Show Figure 2, the DFD diagram)*
"This is our pipeline. It's identical for all four models, so the comparison is fair:
1. **Embedding layer**: turns every word into a 64-number vector that captures its meaning.
2. **Recurrent layer** with 64 units: this is the **only part we change** (SimpleRNN, LSTM, GRU or BiLSTM).
3. **Dropout** of 50% to prevent overfitting.
4. **Dense layer with sigmoid**: outputs a probability. Above 0.5 means positive, below means negative.
We train with the **Adam** optimizer and **binary cross-entropy** loss, batch size 128, and use **early stopping**."

### Slide 6 – The four models (1 min)
"- **SimpleRNN**: the basic recurrent network. Its memory fades, so in long reviews it forgets the beginning. This is called the **vanishing gradient problem**.
- **LSTM**: adds a separate memory called the **cell state**, and three gates (**forget, input and output**) that decide what to remember and what to throw away.
- **GRU**: a simplified LSTM with only two gates (**update and reset**), so it has fewer parameters and is faster.
- **BiLSTM**: two LSTMs, one reading the review forwards and one backwards, so every word gets context from both sides."

### Slide 7 – Results (1.5 min)
*(Show Table 3 and Figure 3)*
"On the 25,000 test reviews, which the models had never seen:
- **SimpleRNN** got **83.12%**, the lowest.
- **LSTM** got **86.76%**, the best accuracy.
- **GRU** got **85.54%**, very close to LSTM.
- **BiLSTM** got **86.62%**, with the best F1-score of 0.867.

*(Show Figure 4, the training curves)* Training accuracy keeps going up to 93%, but validation loss starts rising after epoch 2. This is **overfitting**, and our early stopping caught it and kept the best weights.

*(Show Figure 5, the confusion matrices)* The errors are balanced between positive and negative, so no model is biased towards one class.

*(Show Figure 6, time and parameters)* SimpleRNN is fastest at 95 seconds. LSTM and GRU take about 3 times longer, and BiLSTM is slowest at 537 seconds."

### Slide 8 – Live demo (40 sec)
*(Show Table 4 / demo_predictions.csv)*
"We also tested our own sentences:
- *'This movie was absolutely wonderful'*: all models say positive.
- *'Worst film I have ever seen'*: all say negative.
- *'The plot was predictable **but** the music was great'*: only the gated models understand that the part after 'but' matters. SimpleRNN gets it wrong.
- *'It was **not good**'*: negation is hard. GRU and BiLSTM get it right, but LSTM is on the border at 0.51."

### Slide 9 – Conclusion & future scope (40 sec)
"**Conclusion:** gates matter. LSTM gave the best accuracy, GRU gives almost the same accuracy at lower cost, and BiLSTM is not worth its double training time for this task.
**Future scope:** pretrained GloVe embeddings, an attention mechanism, Transformer models like BERT, and sentiment analysis for Indian languages like Telugu and Hindi.
Thank you. We are happy to take questions."

---

## PART 2 – Viva Questions & Answers

### A. Basics
**Q1. What is sentiment analysis?**
An NLP task that automatically classifies the opinion in a text as positive or negative (sometimes also neutral).

**Q2. What type of problem is this: classification or regression?**
Binary classification, with two classes: positive (1) and negative (0).

**Q3. Why did you use RNNs instead of a normal ANN or CNN?**
Text is sequential and word order matters. An RNN reads words one by one and keeps a hidden state (memory) of the previous words. A normal ANN treats all inputs independently.

**Q4. Which dataset did you use?**
IMDB Large Movie Review dataset (Maas et al., 2011): 50,000 reviews, balanced 50/50. The split was 20k train, 5k validation and 25k test.

**Q5. Why is a balanced dataset important?**
With balanced data, accuracy is a fair metric. With 90% positive data, a model could score 90% just by always predicting "positive".

### B. Preprocessing
**Q6. What is tokenization / integer encoding?**
Splitting text into words and replacing each word with a number (its index in the vocabulary).

**Q7. What is padding? Why 200?**
It makes every review the same length by adding zeros or cutting. 200 covers most reviews (most are under 300 words) and keeps training fast.

**Q8. Why pre-padding (zeros at the start)?**
The RNN reads left to right, so pre-padding keeps the real words at the end, closest to the output. Otherwise the model would read many zeros just before making its decision.

**Q9. Why a vocabulary of 10,000?**
The 10k most frequent words cover almost all of the text. Rare words add noise and parameters.

**Q10. What is an embedding layer?**
A learnable lookup table that converts each word ID into a dense vector (64 numbers). Similar words end up with similar vectors.

**Q11. One-hot encoding vs embedding?**
One-hot vectors are huge (10,000 dimensions), sparse and carry no meaning. Embeddings are small (64 dimensions), dense and capture meaning.

### C. Models
**Q12. What is an unfolded RNN?**
The RNN drawn as a chain, with one copy of the same cell for each time step (200 copies here), all sharing the same weights. Backpropagation through time runs on this unfolded chain.

**Q13. What is the vanishing gradient problem?**
During backpropagation through time, the gradient is multiplied by the same weight matrix at every step. If those values are below 1, it shrinks exponentially, so the early words get almost no learning signal and the RNN "forgets" them.

**Q14. How does LSTM solve it?**
LSTM has a cell state that is updated by addition (c_t = f·c_{t-1} + i·c̃), not repeated multiplication, so gradients can flow back without vanishing. The gates control it:
- Forget gate: what to erase
- Input gate: what new information to write
- Output gate: what to show as the hidden state

**Q15. What is GRU? How is it different from LSTM?**
GRU merges the cell and hidden state and uses 2 gates: update (how much to keep the old memory) and reset (how much of the past to use for the new candidate). It has fewer parameters and is faster, with similar accuracy.

**Q16. What is BiLSTM?**
Two LSTMs: one reads left to right and the other right to left, and their outputs are concatenated. This gives context from both sides of each word.

**Q17. Why didn't BiLSTM win?**
For whole-review classification, the last forward state already summarises the review well, so the backward pass adds little information but doubles the computation.

**Q18. Which model has the most parameters, and why?**
BiLSTM (706k), because it has two LSTMs. For the same number of units: LSTM has 4 weight sets, GRU has 3 and SimpleRNN has 1.

**Q19. Most of the parameters are in which layer?**
The embedding layer: 10,000 × 64 = 640,000. That's why the totals are close.

**Q20. How is your model related to Seq2Seq / encoder–decoder?**
Our model is "many-to-one": a sequence goes in and one label comes out. The recurrent layer works like an encoder, compressing the review into one vector. In Seq2Seq (for example translation), a decoder RNN then generates an output sequence from that vector. We used only the encoder side plus a classifier.

**Q21. What are the types of RNN architectures?**
One-to-one (normal network), one-to-many (image captioning), many-to-one (**our sentiment analysis**), and many-to-many (translation, POS tagging).

### D. Training
**Q22. Why sigmoid at the output?**
It squashes the output to between 0 and 1, so it can be read as the probability of the review being positive. It's the right choice for binary classification.

**Q23. Why binary cross-entropy?**
It's the standard loss for a sigmoid output with two classes, and it heavily penalises predictions that are confident but wrong.

**Q24. Why the Adam optimizer?**
It combines momentum with an adaptive learning rate per parameter (RMSProp), so it converges fast with default settings (lr = 0.001).

**Q25. What is dropout?**
During training it randomly switches off 50% of the neurons so the network can't depend on specific ones. This reduces overfitting.

**Q26. What is early stopping?**
It stops training when validation loss hasn't improved for 2 epochs (patience = 2) and restores the best weights.

**Q27. What is overfitting? Did you observe it?**
Overfitting is when a model memorises the training data and does worse on new data. Yes: training accuracy reached about 93% while validation loss increased after epoch 2.

**Q28. What are epoch, batch size and iteration?**
An epoch is one full pass over the training data. Batch size (128) is the number of samples per weight update. An iteration is one update, and we had 157 per epoch (20,000 ÷ 128).

**Q29. Why a validation set separate from the test set?**
Validation is used during training (for early stopping). The test set is used only once at the end, which gives an unbiased score.

**Q30. Why did you fix the random seed?**
For reproducibility and a fair comparison: all models start from the same random conditions.

### E. Evaluation
**Q31. Explain precision, recall and F1.**
- Precision = TP/(TP+FP): of the reviews predicted positive, how many really were positive.
- Recall = TP/(TP+FN): of the truly positive reviews, how many we found.
- F1 = 2PR/(P+R): the balance between precision and recall.

**Q32. What is a confusion matrix?**
A 2×2 table of TN, FP, FN and TP. For example, LSTM made 1,466 false positives and 1,843 false negatives.

**Q33. Which model is best and why?**
LSTM has the best accuracy (86.76%) and BiLSTM the best F1 (0.867). For practical use, GRU is the best trade-off: 85.54% with less time and fewer parameters.

### F. Limitations & future
**Q34. What are the limitations?**
A single run with one seed, a 200-token cut, no pretrained embeddings, binary classes only, and difficulty with negation and sarcasm.

**Q35. How can you improve accuracy?**
Pretrained embeddings (GloVe/Word2Vec), attention, stacked layers, more epochs with tuning, or BERT (about 93–95% on IMDB).

**Q36. What are the real-world applications?**
E-commerce review analysis, social media brand monitoring, customer-support ticket priority, movie and app ratings, and market sentiment.

**Q37. What tools did you use?**
Python 3.11, TensorFlow/Keras, NumPy, Pandas, scikit-learn, Matplotlib and Seaborn, on a CPU-only laptop.

**Q38. How long does the whole experiment take?**
About 20 minutes on a laptop CPU for all four models.

---

## Who explains what (2-person split)

| Part | Speaker |
|---|---|
| Slides 1–4: intro, problem, dataset | **Harsha Siva Akash Saladi** |
| Slides 5–6: system design, models | **Kishan Bhimavarapu** |
| Slides 7–8: results, live demo | **Harsha Siva Akash Saladi** |
| Slide 9: conclusion and questions | **Kishan Bhimavarapu** |
