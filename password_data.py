import hashlib
import csv
import numpy as np
import pandas as pd
import tensorflow as tf


def load_password_list():
    passwords = []
    for i in range(1, 4):
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


def parse_password_data():
    password_dataset = pd.read_csv("data/password_data/p_data.csv")
    hashes = password_dataset[]

    return x_set, y_set


if __name__ == "__main__":
    parse_password_data()
