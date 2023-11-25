from logistic_regression import logistic_regression
from cross_validation import k_cross_val
from data_loader import load_dataset1

if __name__ == "__main__":
    X, y = load_dataset1()

    print(k_cross_val(logistic_regression, 10, X, y))
