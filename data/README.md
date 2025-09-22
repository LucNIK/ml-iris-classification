# Data Folder – ML Iris Classification

## Description
This folder contains the dataset used for training and evaluating
the Iris classification model.

## Contents
- `iris.csv` (optional) – CSV version of the Iris dataset containing:
  - 4 numerical features: sepal length, sepal width, petal length, petal width
  - 1 target column: species (Setosa, Versicolor, Virginica)

## Notes
- The dataset is loaded directly from `sklearn.datasets.load_iris()` in the scripts.
- If a CSV is added, it can be used for reproducibility or external experiments.
- All data are standardized and ready for model training.

## Usage
- Use this folder as the source for data loading in `scripts/train_model.py` and `scripts/evaluate_model.py`.
