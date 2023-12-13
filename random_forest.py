from data_loader import load_dataset1, load_dataset2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
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

tree1 = DecisionTreeClassifier(random_state=42, max_depth=5, max_features='sqrt', max_leaf_nodes=15, min_samples_leaf=2, min_samples_split=6)
tree2 = DecisionTreeClassifier(random_state=42, max_depth=5, max_features=None, max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=2)

clf_1 = AdaBoostRegressor(tree1, random_state=42)
clf_2 = AdaBoostRegressor(tree2, random_state=42)

params = {
    'n_estimators': [10, 20, 30, 40, 50],
    'learning_rate': [.01, .1, 1],
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
df.to_csv("./results/random_forest_grid_search_data_set1.csv")

grid_search_2.fit(train_x2, train_y2)
df = pd.DataFrame(grid_search_2.cv_results_)
df.to_csv("./results/random_forest_grid_search_data_set2.csv")
