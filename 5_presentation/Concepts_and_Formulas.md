# Concepts & Formulas – Learn the Basics of This Project

A from-zero guide to every concept used in *"A Comparative Study of RNN, LSTM, GRU and BiLSTM Models for Sentiment Analysis of IMDB Movie Reviews"*.
For each topic you get: **simple idea → formula → what it means in our project**.

---

## 1. Deep Learning basics

### 1.1 Neuron
A neuron takes inputs, multiplies each by a weight, adds a bias, and passes the result through an activation function.

$$y = f(w_1x_1 + w_2x_2 + \dots + w_nx_n + b) = f(\mathbf{w}^\top\mathbf{x} + b)$$

- **x** = inputs, **w** = weights (learned), **b** = bias (learned), **f** = activation function.
- **In our project:** the last layer is one neuron with a sigmoid activation that gives the final positive/negative probability.

### 1.2 Activation functions

| Function | Formula | Output range | Where we use it |
|---|---|---|---|
| **Sigmoid** σ | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | 0 to 1 | Output layer (probability) and all LSTM/GRU **gates** |
| **Tanh** | $\tanh(z) = \dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$ | −1 to 1 | Hidden state of RNN/LSTM/GRU |
| ReLU | $\max(0, z)$ | 0 to ∞ | Not used here (common in CNNs) |

**Why sigmoid for gates?** A value between 0 and 1 works like a "percentage open" switch: 0 = block everything, 1 = let everything through.

### 1.3 Training a network

1. **Forward pass:** input → prediction ŷ
2. **Loss:** measure how wrong ŷ is compared with the true label y
3. **Backward pass (backpropagation):** compute the gradient ∂Loss/∂w for every weight using the chain rule
4. **Update:** move each weight a small step against its gradient

**Gradient descent update rule:**
$$w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$
η (eta) = learning rate (we use 0.001).

### 1.4 Key training words

| Term | Meaning | Our value |
|---|---|---|
| **Epoch** | One full pass over all the training data | max 5 |
| **Batch size** | Samples processed before one weight update | 128 |
| **Iteration** | One weight update = one batch | 20,000 / 128 ≈ 157 per epoch |
| **Learning rate** | Size of each update step | 0.001 (Adam default) |
| **Train / Validation / Test** | Learn / tune & early-stop / final unbiased score | 20k / 5k / 25k |

---

## 2. Natural Language Processing (NLP) basics

### 2.1 Sentiment analysis
Classifying the opinion in a text as **positive (1)** or **negative (0)**. It's a **binary classification** task.

### 2.2 Tokenization & integer encoding
Text → words → numbers.
```
"the movie was great"  →  ["the","movie","was","great"]  →  [1, 17, 13, 84]
```
Each number is the word's rank by frequency (1 = most common). We keep only the **top 10,000 words**, and rarer words become the "unknown" token (2).

### 2.3 Padding & truncation
Neural networks need inputs of the same length. We make every review exactly **200 tokens**:
- shorter reviews → add zeros at the **start** (pre-padding)
- longer reviews → cut down to 200 tokens
```
[12, 5, 88]  →  [0, 0, 0, ..., 0, 12, 5, 88]   (length 200)
```

### 2.4 One-hot encoding vs word embedding
- **One-hot:** a word is a 10,000-long vector of zeros with a single 1. It's huge, and every word is equally distant from every other.
- **Embedding:** a word is a short, dense, *learned* vector (64 numbers). Similar words get similar vectors (good ≈ great).

The embedding layer is just a lookup table, a matrix **E** of size 10,000 × 64:
$$\mathbf{x}_t = E[\text{word\_id}_t] \in \mathbb{R}^{64}$$
**Parameters:** 10,000 × 64 = **640,000**, which is most of the parameters in every model.

---

## 3. Recurrent Neural Networks (RNN)

### 3.1 Idea
An RNN reads a sequence **one word at a time** and keeps a **hidden state h_t**, a memory of everything read so far.

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

- $x_t$ = embedding of the current word
- $h_{t-1}$ = memory from the previous word
- The **same weights** $W_x, W_h, b$ are used at every time step (weight sharing).

**Parameters of SimpleRNN (64 units, 64-dim input):**
$$64\times64 + 64\times64 + 64 = 8{,}256$$

### 3.2 Unfolded RNN
"Unfolding" means drawing the loop as a chain, with one copy of the cell per word:
```
x1 → [RNN] → h1 → [RNN] → h2 → ... → [RNN] → h200 → Dense → ŷ
       ↑            ↑                    ↑
      x1           x2                  x200      (same weights in every box)
```
Training on this unfolded chain is called **Backpropagation Through Time (BPTT)**.

### 3.3 Types of RNN architectures

| Type | Example |
|---|---|
| One-to-one | Normal image classification |
| One-to-many | Image captioning |
| **Many-to-one** | **Sentiment analysis (our project)**: 200 words in → 1 label out |
| Many-to-many | Machine translation, POS tagging |

### 3.4 Vanishing gradient problem ⚠️
In BPTT, the gradient reaching word 1 is a **product of 200 terms**:
$$\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_{200}} \prod_{t=2}^{200} \frac{\partial h_t}{\partial h_{t-1}}$$
If each term is less than 1 (e.g. 0.9), then 0.9²⁰⁰ ≈ 0.0000000007, so the gradient **vanishes**. The early words get no learning signal and the RNN **forgets the beginning of long reviews**.
(If the terms are greater than 1, the gradient **explodes** instead.)

**This is exactly why SimpleRNN scored lowest (83.12%) in our project.**

---

## 4. LSTM – Long Short-Term Memory

### 4.1 Idea
LSTM adds a separate long-term memory called the **cell state $c_t$**, plus **3 gates** that decide what to forget, what to write and what to output.

### 4.2 Formulas
$[h_{t-1}, x_t]$ means the two vectors joined together, and ⊙ means element-wise multiplication.

| Step | Formula | Meaning |
|---|---|---|
| Forget gate | $f_t = \sigma(W_f[h_{t-1},x_t] + b_f)$ | What to **erase** from memory (0 = erase, 1 = keep) |
| Input gate | $i_t = \sigma(W_i[h_{t-1},x_t] + b_i)$ | How much new information to **write** |
| Candidate | $\tilde{c}_t = \tanh(W_c[h_{t-1},x_t] + b_c)$ | The new information itself |
| **Cell update** | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | Keep part of the old memory and add the new |
| Output gate | $o_t = \sigma(W_o[h_{t-1},x_t] + b_o)$ | What part of the memory to **show** |
| Hidden state | $h_t = o_t \odot \tanh(c_t)$ | Output at this step |

### 4.3 Why it fixes vanishing gradients
The cell state is updated by **addition** ($c_t = f_t c_{t-1} + \dots$), not repeated matrix multiplication. When $f_t ≈ 1$, the gradient flows back almost unchanged, like a highway.

### 4.4 Parameters (64 units, 64-dim input)
4 weight blocks (f, i, o, c):
$$4 \times (64\times(64+64) + 64) = 33{,}024$$
Total model = 640,000 (embedding) + 33,024 + 65 (dense) = **673,089** ✓ (matches our results table)

**In our project:** LSTM had the **best accuracy, 86.76%**.

---

## 5. GRU – Gated Recurrent Unit

### 5.1 Idea
A simpler LSTM: there is no separate cell state, and only **2 gates**.

### 5.2 Formulas
| Step | Formula | Meaning |
|---|---|---|
| Update gate | $z_t = \sigma(W_z[h_{t-1},x_t] + b_z)$ | How much to replace the old memory with new |
| Reset gate | $r_t = \sigma(W_r[h_{t-1},x_t] + b_r)$ | How much of the past to use when making the candidate |
| Candidate | $\tilde{h}_t = \tanh(W[r_t \odot h_{t-1}, x_t] + b)$ | New proposed memory |
| **Hidden state** | $h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t$ | A mix of the old and the new |

### 5.3 LSTM vs GRU

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Memory | Cell state + hidden state | Hidden state only |
| Weight blocks | 4 | 3 → **fewer parameters** |
| Speed | Slower | **Faster** |
| Our accuracy | 86.76% | 85.54% |
| Our training time | 296 s | 272 s |

---

## 6. Bidirectional LSTM (BiLSTM)

Two LSTMs: one reads **left → right**, the other reads **right → left**. Their final states are joined:
$$h = [\overrightarrow{h}_{200} \; ; \; \overleftarrow{h}_{1}] \in \mathbb{R}^{128}$$
- **Benefit:** every word sees context from both sides.
- **Cost:** 2× the parameters in the recurrent layer and about 2× the time.
- **In our project:** best F1 (0.867) and recall, but no accuracy gain over LSTM, and 1.8× slower.

---

## 7. Seq2Seq and Encoder–Decoder (syllabus link)

- **Encoder:** an RNN/LSTM/GRU reads the input sequence and compresses it into a **context vector** (its final hidden state).
- **Decoder:** another RNN generates the output sequence from that vector, one token at a time.
- Used in machine translation ("I am happy" → "मैं खुश हूँ"), chatbots and summarisation.

**Link to our project:** our recurrent layer is the **encoder** (review → one vector). Instead of a decoder that generates a sequence, we use a **Dense + sigmoid** layer that outputs one label.

---

## 8. Output, loss and optimiser

### 8.1 Output layer
$$\hat{y} = \sigma(\mathbf{w}^\top \mathbf{h} + b) \qquad \text{Prediction} = \begin{cases}\text{Positive} & \hat{y} \ge 0.5\\ \text{Negative} & \hat{y} < 0.5\end{cases}$$

### 8.2 Binary cross-entropy loss
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big]$$
**Example:** if the true label is y = 1 and the model says ŷ = 0.9, the loss is −log(0.9) = 0.105 (small). If ŷ = 0.1, the loss is −log(0.1) = 2.303 (large). Confident wrong answers are punished heavily.

### 8.3 Adam optimiser
Adam keeps a running average of the gradients (momentum, $m_t$) and of the squared gradients ($v_t$):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad \hat{v}_t = \frac{v_t}{1-\beta_2^t},\qquad w \leftarrow w - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$
Defaults: η = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁷. Each weight gets its **own adaptive step size**.

---

## 9. Regularisation (fighting overfitting)

**Overfitting** = the model memorises the training data. Training accuracy goes up, but validation accuracy goes down or validation loss goes up.
**In our project:** training accuracy reached about 93% while validation loss rose after epoch 2 (see `2_results/fig1_training_curves.png`).

| Technique | How it works | Our setting |
|---|---|---|
| **Dropout** | During training, randomly set a fraction p of the neurons to 0, so the network can't depend on any single neuron | p = 0.5 |
| **Early stopping** | Stop when validation loss hasn't improved for *k* epochs and restore the best weights | monitor = val_loss, patience = 2 |

---

## 10. Evaluation metrics

### 10.1 Confusion matrix
|  | **Predicted Negative** | **Predicted Positive** |
|---|---|---|
| **Actual Negative** | TN (True Negative) ✅ | FP (False Positive) ❌ |
| **Actual Positive** | FN (False Negative) ❌ | TP (True Positive) ✅ |

### 10.2 Formulas

| Metric | Formula | Question it answers |
|---|---|---|
| **Accuracy** | $\dfrac{TP+TN}{TP+TN+FP+FN}$ | How many did we get right overall? |
| **Precision** | $\dfrac{TP}{TP+FP}$ | When we say "positive", how often are we right? |
| **Recall** | $\dfrac{TP}{TP+FN}$ | Of all the real positives, how many did we find? |
| **F1-score** | $\dfrac{2 \cdot P \cdot R}{P + R}$ | The balance of precision and recall |

### 10.3 Worked example: our LSTM
From the LSTM confusion matrix: TN = 11,034, FP = 1,466, FN = 1,843, TP = 10,657.
- Accuracy = (10,657 + 11,034) / 25,000 = 21,691 / 25,000 = **0.8676** ✓
- Precision = 10,657 / (10,657 + 1,466) = 10,657 / 12,123 = **0.8791** ✓
- Recall = 10,657 / (10,657 + 1,843) = 10,657 / 12,500 = **0.8526** ✓
- F1 = 2 × 0.8791 × 0.8526 / (0.8791 + 0.8526) = **0.8656** ✓

---

## 11. Quick revision sheet (one-liners)

- **RNN:** $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$. It has memory but forgets over long sequences.
- **Vanishing gradient:** a product of many numbers below 1 goes to 0, so early words aren't learned.
- **LSTM:** cell state plus forget/input/output gates. Its additive memory solves vanishing gradients.
- **GRU:** update and reset gates. Fewer parameters, faster, almost the same accuracy.
- **BiLSTM:** forward + backward LSTM. Context from both sides at 2× the cost.
- **Embedding:** word ID → learned dense vector (10,000 × 64).
- **Sigmoid:** 1/(1+e⁻ᶻ) gives a probability between 0 and 1.
- **BCE loss:** −[y log ŷ + (1−y) log(1−ŷ)].
- **Adam:** momentum + adaptive learning rate.
- **Dropout 0.5 and early stopping:** fight overfitting.
- **Accuracy / Precision / Recall / F1:** see section 10.
- **Our winner:** LSTM 86.76% accuracy. **Best value:** GRU. **Not worth the cost:** BiLSTM.
