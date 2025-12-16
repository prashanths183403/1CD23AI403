import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

def AlexNet(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(96, (11,11), strides=4, activation='relu'),
        MaxPooling2D((3,3), strides=2),

        Conv2D(256, (5,5), padding='same', activation='relu'),
        MaxPooling2D((3,3), strides=2),

        Conv2D(384, (3,3), padding='same', activation='relu'),
        Conv2D(384, (3,3), padding='same', activation='relu'),
        Conv2D(256, (3,3), padding='same', activation='relu'),
        MaxPooling2D((3,3), strides=2),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = AlexNet((224,224,3), 1000)
model.summary()
