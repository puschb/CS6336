import numpy as np


# just loads the data, doesn't perform random sammpling for splits
def load_dataset1():
    data_set_xs = []
    data_set_ys = []
    with open("./data/project3_dataset1.txt", "r", encoding="utf-8") as f:
        for line in f:
            vals = line.split("\t")
            data_set_xs.append(vals[0:-1])
            data_set_ys.append(vals[-1])

    data_set_xs_numpy = np.array(data_set_xs, dtype="float64")
    data_set_ys_numpy = np.array(data_set_ys, dtype="float64")
    return data_set_xs_numpy, data_set_ys_numpy


# just loads the data, doesn't perform random sammpling for splits
# converts present/absent to 1/0
def load_dataset2():
    data_set_xs = []
    data_set_ys = []
    with open("./data/project3_dataset2.txt", "r", encoding="utf-8") as f:
        for line in f:
            vals = line.split("\t")
            vals[4] = 1 if vals[4] == "Present" else 0
            data_set_xs.append(vals[0:-1])
            data_set_ys.append(vals[-1])

    data_set_xs_numpy = np.array(data_set_xs, dtype="float64")
    data_set_ys_numpy = np.array(data_set_ys, dtype="float64")
    return data_set_xs_numpy, data_set_ys_numpy
