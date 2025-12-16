import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# LOAD DATASET
# -------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# -------------------------------
# DISPLAY SAMPLE IMAGES
# -------------------------------
plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis("off")
plt.show()

# -------------------------------
# MODEL
# -------------------------------
model = Sequential([
    Input(shape=(32,32,3)),

    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, activation='relu'),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.summary()

# -------------------------------
# COMPILE
# -------------------------------
model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# TRAIN
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    verbose=2
)

# -------------------------------
# EVALUATE
# -------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

# -------------------------------
# PLOTS
# -------------------------------
epochs = range(len(history.history['accuracy']))

plt.plot(epochs, history.history['accuracy'], label="Train Acc")
plt.plot(epochs, history.history['val_accuracy'], label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(epochs, history.history['loss'], label="Train Loss")
plt.plot(epochs, history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.show()

# -------------------------------
# RANDOM TEST IMAGE PREDICTION
# -------------------------------
idx = random.randint(0, len(x_test)-1)
img = x_test[idx]
true_label = y_test[idx][0]

pred = model.predict(img[np.newaxis, ...], verbose=0)
predicted_label = np.argmax(pred)

print("True:", class_names[true_label])
print("Predicted:", class_names[predicted_label])

plt.imshow(img)
plt.title(f"Predicted: {class_names[predicted_label]}")
plt.axis("off")
plt.show()

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("cifar10_cnn.keras")
print("Model saved successfully")
