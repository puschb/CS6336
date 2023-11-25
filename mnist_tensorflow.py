import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)
import numpy as np

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # create sequential model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(28, 28)))

    # add rnn layers
    model.add(tf.keras.layers.GRU(128, return_sequences=True, activation="relu"))
    model.add(tf.keras.layers.GRU(256, return_sequences=True, activation="relu"))
    model.add(tf.keras.layers.SimpleRNN(128, return_sequences=False, activation="relu"))

    # add dense layer with number of possibilities
    model.add(tf.keras.layers.Dense(10))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=64, verbose=2)
