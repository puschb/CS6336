import os
import pandas as pd
import numpy as np
import json


def sort_best(file):
    """
    sort the data from the current file by accuracy and separate
    the parameters into their own columns
    """
    data = pd.read_csv(file)[
        [
            "params",
            "mean_test_accuracy",
            "mean_test_precision",
            "mean_test_recall",
            "mean_test_f1",
            "mean_test_roc_auc",
        ]
    ]

    # make a dict mapping from current colname to new colname
    colnames = {
        item: item.replace("mean_test_", "")
        for item in list(data.columns.values.tolist())
    }

    data = data.rename(columns=colnames)
    param_list = list(
        json.loads(
            data.iloc[0]["params"].replace("'", '"').replace("None", "null")
        ).keys()
    )

    # convert to dict for iterative param list processing
    data_dict = data.to_dict(orient="records")
    for row in data_dict:
        # replace single quotes with double quotes and None values with null
        temp = json.loads(row["params"].replace("'", '"').replace("None", "null"))
        row.update(temp)
        row.pop("params")

    # convert the dataframe back from dict
    data = pd.DataFrame.from_dict(data_dict)
    data = data.sort_values(
        by=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
    )
    # get the revised filename
    fname = (
        file.split("/")[-1].split("_")[0]
        + "_"
        + file.split(".")[-2].split("_")[-1]
        + "_culled.csv"
    )
    # get the path to the revised file
    path = "/".join(file.split("/")[:-1])
    # folder must already exist
    data.to_csv(path + "/culled/" + fname)
    # print a message to the console if the correlation
    # coefficient between accuracy and parameter is greater than 0.8
    for p in param_list:
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if data[p].dtypes == np.int64 or data[p].dtypes == np.float64:
                if abs((data[m].corr(data[p]))) > 0.8:
                    cor = data[m].corr(data[p])
                    print(f"Correlation between {m} and {p} for {fname} is {cor}")


def find_best(file):
    data = pd.read_csv(file)[
        [
            "params",
            "mean_test_accuracy",
            "mean_test_precision",
            "mean_test_recall",
            "mean_test_f1",
            "mean_test_roc_auc",
        ]
    ]

    # data["average"] = data.mean(axis=1)
    maxes = data.max(axis=0)
    indices = data.index[
        data["mean_test_accuracy"] == maxes["mean_test_accuracy"]
    ].tolist()

    best_classifier = data.iloc[indices]
    best_classifier = best_classifier.sort_values(
        by=[
            "mean_test_precision",
            "mean_test_recall",
            "mean_test_f1",
            "mean_test_roc_auc",
        ],
        ascending=False,
    )
    return best_classifier.iloc[0]


def get_bests(results_path="./results/"):
    # check file extension
    files = [
        f
        for f in os.listdir("./results/")
        if ".csv" == f[-4:]
        and any(
            word in f
            for word in ["decision", "k_neighbor", "logistic", "random_forest", "svm"]
        )
    ]
    bests = []
    for f in files:
        best = find_best(results_path + f)
        best["name"] = f
        bests.append(best)

    result = pd.DataFrame(bests)
    result = result.sort_values(by="name").reset_index()
    # reorder columns so that names come first
    cols = result.columns.tolist()[-1:] + result.columns.tolist()[:-1]
    result = result[cols]
    result.to_csv(results_path + "best/results.csv")


def decipher_results(results_path="./results/"):
    # check file extension
    files = [
        f
        for f in os.listdir("./results/")
        if ".csv" == f[-4:]
        and any(
            word in f
            for word in ["decision", "k_neighbor", "logistic", "random_forest", "svm"]
        )
    ]

    for f in files:
        sort_best(results_path + f)


if __name__ == "__main__":
    get_bests()
    decipher_results()
