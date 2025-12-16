import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

# -----------------------------
# DATA
# -----------------------------
text = "the beautiful girl is intelligent"
chars = sorted(set(text))

char2idx = {c:i for i,c in enumerate(chars)}
idx2char = {i:c for c,i in char2idx.items()}

vocab_size = len(chars)
seq_len = 5

# -----------------------------
# DATA PREPARATION
# -----------------------------
X, y = [], []
for i in range(len(text) - seq_len):
    X.append([char2idx[c] for c in text[i:i+seq_len]])
    y.append(char2idx[text[i+seq_len]])

X = np.array(X)
y = np.array(y)

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(100).batch(8)

# -----------------------------
# MODEL
# -----------------------------
model = Sequential([
    Input(shape=(seq_len,)),
    Embedding(vocab_size, 16),
    LSTM(64),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

model.summary()

# -----------------------------
# TRAINING
# -----------------------------
model.fit(dataset, epochs=100, verbose=0)

# -----------------------------
# TEXT GENERATION
# -----------------------------
def generate_text(seed, length=30):
    output = seed
    for _ in range(length):
        x = np.array([[char2idx[c] for c in output[-seq_len:]]])
        preds = model.predict(x, verbose=0)[0]
        next_char = idx2char[np.argmax(preds)]
        output += next_char
    return output

print("Generated Text:")
print(generate_text("the b"))
