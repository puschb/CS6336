import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import keras
from password_data import load_password_data, tvt_split, Autoencoder
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

tf.get_logger().setLevel("FATAL")
tf.autograph.set_verbosity(1)


@keras.saving.register_keras_serializable()
class Hasher(Model):
    def __init__(self):
        super(Hasher, self).__init__()
        self.shape = (1, 96)
        self.o_shape = (1, 128)
        self.hidden = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(tf.math.reduce_prod(self.o_shape), activation="sigmoid"),
                layers.Reshape(self.o_shape),
            ]
        )

    def call(self, x):
        prediction = self.hidden(x)
        return prediction


@keras.saving.register_keras_serializable()
class Reverser(Model):
    def __init__(self, f_len=128, o_len=96):
        super(Reverser, self).__init__()
        self.shape = (1, f_len)
        self.o_shape = (1, o_len)
        self.hidden = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dense(tf.math.reduce_prod(self.o_shape), activation="sigmoid"),
                layers.Reshape(self.o_shape),
            ]
        )

    def call(self, x):
        prediction = self.hidden(x)
        return prediction


def train_models():
    # builder = tfds.builder("rock_you")
    # x_train = builder.as_dataset(split="train[:50%]")
    # x_test = builder.as_dataset(split="train[50%:]")
    pw, h, epw = load_password_data()
    train_p, val_p, test_p = tvt_split(pw)
    train_h, val_h, test_h = tvt_split(h)
    train_epw, val_epw, test_epw = tvt_split(epw)
    print(train_h)

    hasher = Hasher()
    hasher.compile(optimizer="adam", loss=losses.MeanSquaredError())
    hasher.fit(train_epw, train_h, epochs=10, shuffle=True)

    hasher.save("models/hasher.keras")

    reverser = Reverser()
    reverser.compile(optimizer="adam", loss=losses.MeanSquaredError())
    reverser.fit(train_h, train_epw, epochs=10, shuffle=True)

    reverser.save("models/reverser.keras")

    naive_reverser = Reverser(128, 192)
    naive_reverser.compile(optimizer="adam", loss=losses.MeanSquaredError())
    naive_reverser.fit(train_h, train_p, epochs=10, shuffle=True)

    reverser.save("models/naive_reverser.keras")

    hasher.evaluate(val_epw, val_h)
    reverser.evaluate(val_h, val_epw)
    naive_reverser.evaluate(val_h, val_p)


def test_against_random_guessing():
    # set up data
    pw, h, epw = load_password_data()
    train_p, val_p, test_p = tvt_split(pw)
    train_h, val_h, test_h = tvt_split(h)
    train_epw, val_epw, test_epw = tvt_split(epw)

    # generate random outputs
    random_pw = np.random.randint(2, size=val_p.shape)
    random_hash = np.random.randint(2, size=val_h.shape)
    random_epw = np.random.random_sample(val_epw.shape)

    # load models
    hasher = tf.keras.models.load_model("./models/hasher.keras")
    reverser = tf.keras.models.load_model("./models/reverser.keras")
    naive_reverser = tf.keras.models.load_model("./models/naive_reverser.keras")

    # evaluate models
    hasher_loss = hasher.evaluate(val_epw, val_h)
    reverser_loss = reverser.evaluate(val_h, val_epw)
    # naive_loss = naive_reverser.evaluate(val_h, val_p)

    # evaluate random guessing
    r_pw_loss = mean_squared_error(random_pw, val_p)
    r_hash_loss = mean_squared_error(random_hash, val_h)
    r_epw_loss = mean_squared_error(random_epw, val_epw)

    # compare against random outputs
    hasher_score = r_hash_loss - hasher_loss
    reverser_score = r_epw_loss - reverser_loss
    # naive_reverser_score = r_pw_loss - naive_loss

    ae = tf.keras.models.load_model("./models/autoencoder.keras")

    r_hash_acc = ((random_hash == val_h).flatten()).sum() / len(val_h.flatten())
    r_pw_acc = ((random_pw == val_p).flatten()).sum() / len(val_p.flatten())

    reverser_epw = reverser.predict(val_h).reshape((val_epw.shape))
    reverser_pw = []
    print(reverser_epw)
    for row in tqdm(reverser_epw):
        temp = []
        for i in range(0, len(row), 4):
            print(row[i : i + 4])
            temp.append(ae.get_decoded(row[i : i + 4].reshape((1, 4))))
        temp = temp.flatten()
        temp = np.round(temp, decimals=0)
        reverser_pw.append(temp)

    print(reverser_pw)

    # print out comparisons
    print(f"hasher MSE: {hasher_loss}")
    print(f"reverser MSE: {reverser_loss}")
    print(f"random MSE - hasher MSE: {hasher_score}")
    print(f"{r_hash_loss / hasher_loss} times better")
    print(f"random MSE - reverser MSE: {reverser_score}")
    print(f"{r_epw_loss / reverser_loss} times better")
    return hasher_score, reverser_score


if __name__ == "__main__":
    test_against_random_guessing()
