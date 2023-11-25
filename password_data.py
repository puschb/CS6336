import hashlib
import csv
import numpy as np


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
    dataset = get_password_data()
    x = dataset[:, 0]
    y = dataset[:, 1]
    x_set = []
    for item in x:
        x_set.append([ord(c) for c in item])
    y_set = [[int(s[i : i + 1], 16) for i in range(0, len(s), 2)] for s in y]
    return x_set, y_set


if __name__ == "__main__":
    parse_password_data()
