import hashlib
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


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


def parse_password_data():
    password_dataset = get_password_data()

    # convert hashes to ascii
    hashes = [[ord(c) for c in s] for s in password_dataset[:, 1]]

    # pad each password in the dataset to 23 characters
    password = [
        [ord(s[c]) if c < len(s) else ord(" ") for c in range(24)]
        for s in password_dataset[:, 0]
    ]

    return np.array(password, dtype="int32"), np.array(hashes, dtype="int32")


if __name__ == "__main__":
    write_hashed_data()
