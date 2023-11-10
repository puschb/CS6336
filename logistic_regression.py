from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from data_loader import load_dataset1


def logistic_regression(X, y):
    clf = LogisticRegression(max_iter=10000, random_state=0)
    clf.fit(X, y)

    return clf


if __name__ == "__main__":
    X, y = load_dataset1()
    print(logistic_regression(X, y).score(X, y))
