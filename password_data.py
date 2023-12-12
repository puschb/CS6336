import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hashlib
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow import keras
from tqdm import tqdm


def load_password_list():
    passwords = []
    for i in range(1, 11):
        with open(f"data/password_data/passwords_{i}.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                passwords.append(row[0])
    return passwords


def hash_passwords(passwords):
    hashes = []
    for p in passwords:
        hashes.append(hashlib.md5(p.encode("UTF-8")).hexdigest())
    return hashes


def write_hashed_data():
    passwords = load_password_list()
    hashes = hash_passwords(passwords)

    with open("data/password_data/p_data.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        for p, h in zip(passwords, hashes):
            writer.writerow([p, h])


def get_password_data():
    dataset = []
    with open("data/password_data/p_data.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            dataset.append(row)

    return np.asarray(dataset)


def str_arr_to_bin_arr(dataset, standard_length=24):
    dataset = [
        [ord(s[c]) if c < len(s) else ord(" ") for c in range(standard_length)]
        for s in dataset
    ]

    return np.reshape(
        np.array(
            [[list(bin(c)[2:].zfill(8)) for c in d] for d in dataset], dtype="int8"
        ),
        (-1, standard_length * 8),
    )


def hex_arr_to_bin_arr(dataset, standard_length=16):
    dataset = [np.binary_repr(int(s, 16), width=8 * standard_length) for s in dataset]
    dataset = [[int(i) for i in list(item)] for item in dataset]
    return dataset


def parse_password_data():
    password_dataset = get_password_data()

    password = str_arr_to_bin_arr(password_dataset[:, 0])
    hashes = hex_arr_to_bin_arr(password_dataset[:, 1], standard_length=16)

    return np.array(password, dtype="int8"), np.array(hashes, dtype="int8")


def preprocess_data():
    pass


@keras.saving.register_keras_serializable()
class Autoencoder(Model):
    def __init__(self, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = 4
        self.shape = shape
        self.encoder = tf.keras.Sequential(
            [layers.Flatten(), layers.Dense(self.latent_dim, activation="relu")]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(tf.math.reduce_prod(shape), activation="sigmoid"),
                layers.Reshape(shape),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(x)
        return decoded

    def get_encoded(self, x):
        encoded = self.encoder(x)
        return encoded

    def get_decoded(self, x):
        decoded = self.decoder(x)
        return decoded


def train_input_encoder():
    characters = list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~`!@#$%^&*()_-+={[]}|\:;"
        "<,>.?/"
    )
    characters = str_arr_to_bin_arr(characters, standard_length=1)
    shape = characters.shape[1:]
    ae = Autoencoder(shape)
    ae.compile(optimizer="adam", loss=losses.MeanSquaredError())

    ae.fit(characters, characters, epochs=20000, shuffle=True)

    ae.save("models/autoencoder.keras")


def write_file(data, filename):
    with open(f"data/password_data/{filename}", "w") as f:
        writer = csv.writer(f, delimiter=",")
        for row in data:
            writer.writerow(row)


def encode_dataset():
    ae = tf.keras.models.load_model("models/autoencoder.keras")
    pw, hash = parse_password_data()
    epw = []
    # this loop takes a very long time to execute
    # because of all of the copies that occur inside of it
    # unfortunately, there is no way to resolve this
    for word in tqdm(pw):
        encoded_word = []
        for i in range(0, len(word), 8):
            encoded_word.append(
                ae.get_encoded(word[i : i + 8].reshape(1, 8)).numpy().tolist()
            )
        epw.append(np.asarray(encoded_word).flatten())

    epw = np.asarray(epw)

    write_file(pw, "binary_passwords.csv")
    write_file(hash, "binary_hashes.csv")
    write_file(epw, "encoded_passwords.csv")


def load_password_data():
    password = np.genfromtxt("./data/password_data/binary_passwords.csv", delimiter=",")
    hashes = np.genfromtxt("./data/password_data/binary_hashes.csv", delimiter=",")
    encoded_passwords = np.genfromtxt(
        "./data/password_data/encoded_passwords.csv", delimiter=","
    )

    return password, hashes, encoded_passwords


def tvt_split(data):
    train = data[0 : int(len(data) * 0.8)]
    validation = data[int(len(data) * 0.8) : int(len(data) * 0.9)]
    test = data[int(len(data) * 0.9) :]
    return train, validation, test


if __name__ == "__main__":
    tf.get_logger().setLevel("FATAL")
    tf.autograph.set_verbosity(1)
    load_password_data()
