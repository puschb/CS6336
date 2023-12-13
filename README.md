# Classification Project

Authors: Mason Barnes, William Bradford, Benjamin Pusch

## Datasets 1 & 2

The models trained on these datasets were trained using the files svm.py,
logistic_regression.py, random_forest.py, knn.py, and decision_tree.py. Running
these .py files produces a CSV file corresponding to each .py file that we used
for data analysis. This is stored in the results folder. The data analysis was
done using data_analysis.py. You may run this file if you wish to conduct the
same analysis that we did for the project, however the results should be
properly stored in the distributed CSV files found in "results/culled" already.

## MNIST

The grid search used for the data analysis of the MNIST neural network was done
in the file "mnist_torch.py". Running the main method of this file will execute
the grid search once again. Running the train_model method will train a model
with the given input parameters. This has defaults and can be used as
train_model(). The results of this grid search were stored in the CSV file
"mnist_search.csv," which can be found in the results folder.

## Extra Credit

The extra credit was done using the provided datasets and the password_tf.py
file. The password data processing was done using password_data.py. If you wish
to re-process this data, you may use the methods found there to do so, however
it has all be cleanly preprocessed and included in this zip file.

_Note_: The password_data.py file does not do anything on its own. You will need
to modify the main method or import methods in a separate script to run any of
the methods inside.

If you wish to retrain the models, this can be done using the train_models()
method, however the main method of that file will load the pretrained models and
test them against random guesses using the MSE metric.

The accuracy metric was still unimplemented as of the end of the semester.

## Included File List

#### Python files:

- svm.py
- logistic_regression.py
- random_forest.py
- decision_tree.py
- knn.py
- data_analysis.py
- data_loader.py
- mnist_torch.py
- password_data.py
- password_tf.py

#### CSV files:

- results/decision_tree_grid_search_data_set(1 and 2).csv
- results/k_neighbor_grid_search_data_set(1 and 2).csv
- results/logistic_regression_grid_search_data_set(1 and 2).csv
- results/random_forest_grid_search_data_set(1 and 2).csv
- results/svm_classifier_grid_search_data_set(1 and 2).csv
- results/mnist_search.csv
- results/culled/decision_set(1 and 2)\_culled.csv
- results/culled/k_set(1 and 2)\_culled.csv
- results/culled/logistic_set(1 and 2)\_culled.csv
- results/culled/random_set(1 and 2)\_culled.csv
- results/culled/svm\_(1 and 2)\_culled.csv
- results/best/results.csv
  - stores best results overall

#### Model files:

- results/mnist_models/\*.pt
- models/\*.keras
