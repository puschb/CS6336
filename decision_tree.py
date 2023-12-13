from data_loader import load_dataset1, load_dataset2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#load and pre-process data
#mild imbalance in classes but should be fine
x1, y1 = load_dataset1()
x2, y2 = load_dataset2()
scalar = StandardScaler(copy=False)
scalar.fit(x1)
scalar.transform(x1)
scalar.fit(x2)

#binary var
scalar.mean_[4] = 0
scalar.var_[4] = 1
scalar.scale_[4] = 1
scalar.transform(x2)

#generate train and test split - NOTE: we do not use our test split at all until we have decided that we have our best model (which we will decide based on the validation
# loss)
train_x1, test_x1, train_y1, test_y1 = train_test_split(x1,y1, test_size=.2, random_state=42)
train_x2, test_x2, train_y2, test_y2 = train_test_split(x2,y2,test_size=.2, random_state=42)

clf_1 = DecisionTreeClassifier(random_state=42)
clf_2 = DecisionTreeClassifier(random_state=42)

params = {
    "max_depth": np.linspace(5,40,8,dtype='int'),
    "min_samples_split": np.linspace(2,15,7,dtype='int'),
    "min_samples_leaf": [1, 2, 3, 4,5,6,7],
    "max_features": [None, "sqrt", "log2"],
    "max_leaf_nodes": np.linspace(5,40,8,dtype='int'),
}

grid_search_1 = GridSearchCV(
    clf_1,
    params,
    cv=10,
    scoring=("accuracy", "f1", "precision", "recall", "roc_auc"),
    refit=False,
    verbose=2,
    n_jobs=-1,
    error_score=-1,
    return_train_score=True
)
grid_search_2 = GridSearchCV(
    clf_2,
    params,
    cv=10,
    scoring=("accuracy", "f1", "precision", "recall", "roc_auc"),
    refit=False,
    verbose=2,
    n_jobs=-1,
    error_score=-1,
    return_train_score=True
)

# run grid search, write results to csv file
grid_search_1.fit(train_x1, train_y1)
df = pd.DataFrame(grid_search_1.cv_results_)
df.to_csv("./results/decision_tree_grid_search_data_set1.csv")

grid_search_2.fit(train_x2, train_y2)
df = pd.DataFrame(grid_search_2.cv_results_)
df.to_csv("./results/decision_tree_grid_search_data_set2.csv")
