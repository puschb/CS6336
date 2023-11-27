import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from password_data import parse_password_data
from sklearn.preprocessing import normalize
import numpy as np

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

if __name__ == "__main__":
    # builder = tfds.builder("rock_you")
    # x_train = builder.as_dataset(split="train[:50%]")
    # x_test = builder.as_dataset(split="train[50%:]")
    pword, hash = parse_password_data()

    plen, hlen = len(pword[0]), len(hash[0])

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(192, 1)))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(hlen))

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # save model at each epoch
    model_save = tf.keras.callbacks.ModelCheckpoint(
        "models/checkpoint/",
        save_weights_only=True,
        monitor="val_loss",
        save_freq="epoch",
        mode="min",
        save_best_only=True,
    )

    model.compile(optimizer="adam", loss=loss)

    model.fit(pword, hash, batch_size=64, epochs=5, verbose=2, callbacks=[model_save])
    model.evaluate(pword, hash, batch_size=64, verbose=2)

    predictions = model.predict(pword)
    print(predictions[0])
    print(hash[0])
