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


def parse_password_data():
    password_dataset = get_password_data()

    password = str_arr_to_bin_arr(password_dataset[:, 0])
    hashes = str_arr_to_bin_arr(password_dataset[:, 1], standard_length=32)

    return np.array(password, dtype="int8"), np.array(hashes, dtype="int8")


if __name__ == "__main__":
    write_hashed_data()
