from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pandas as pd
import numpy as np


X = pd.read_csv('data/X.csv')
y = pd.read_csv('data/y.csv')['class']
X = X
y = y.astype(int)
X = ((X / 255.) - .5) * 2

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

# optional to free up some memory by deleting non-used arrays:
del X_temp, y_temp, X, y


def build_ann(x_train, y_train, num_classes):
    """
    """

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    model = Sequential()
    model.add(Dense(units=50, activation='sigmoid', input_dim=len(x_train.columns)))
    model.add(Dense(units=50, activation='sigmoid'))
    model.add(Dense(units=10, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=tf.keras.metrics.AUC())
    model.fit(x_train, y_train, epochs=100, batch_size=100)
    model.save('ann_model')


# build_ann(X_train, y_train, 10)
model = load_model('ann_model')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
score = model.evaluate(X_test, y_test)
print(("\n\nModel Accuracy: %.2f%%" % (score[1]*100)))