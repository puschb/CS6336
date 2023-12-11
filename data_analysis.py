import os
import pandas as pd
import numpy as np


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


def analyze_results(results_path="./results/"):
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


if __name__ == "__main__":
    analyze_results()
