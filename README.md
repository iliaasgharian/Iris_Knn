# Iris Flower Classification Using K-Nearest Neighbors (KNN)

## Project Overview

This project implements a K-Nearest Neighbors (KNN) classification
algorithm to classify Iris flowers into three different species using
two groups of features:

-   **Sepal Features:** Sepal Length and Sepal Width
-   **Petal Features:** Petal Length and Petal Width

The main objective is to compare the classification performance of these
two feature groups and find the optimal value of K for each group using
the **Macro F1-Score**.

## Dataset

The project uses the Iris dataset (`iris.csv`), which contains
measurements of three Iris flower species:

-   Setosa
-   Versicolor
-   Virginica

The dataset includes four numerical features:

  Feature          Description
  ---------------- --------------
  `sepal.length`   Sepal length
  `sepal.width`    Sepal width
  `petal.length`   Petal length
  `petal.width`    Petal width

## Methodology

The project follows these steps:

### 1. Data Loading

The Iris dataset is loaded using Pandas.

### 2. Feature Selection

Two separate feature groups are defined:

-   Sepal: `sepal.length`, `sepal.width`
-   Petal: `petal.length`, `petal.width`

### 3. Train-Test Split

The dataset is divided into:

-   80% Training Data
-   20% Testing Data

The `stratify` parameter is used to maintain the class distribution in
both sets.

### 4. Feature Standardization

`StandardScaler` is used to standardize the numerical features.

Standardization is included inside a Scikit-learn Pipeline to prevent
data leakage during cross-validation.

### 5. Optimal K Selection

`GridSearchCV` is used to find the best value of K.

The algorithm tests K values from 1 to 30 using 5-Fold Stratified Cross
Validation.

The optimization metric is **Macro F1-Score**.

The K value with the highest average cross-validation F1-Score is
selected as the optimal K.

### 6. Model Evaluation

The best model is evaluated using the independent test set.

The following evaluation metrics are calculated:

-   Accuracy
-   Macro F1-Score
-   Confusion Matrix
-   Classification Report

In addition, a results table is built for **every K from 1 to 30** (for
both Sepal and Petal) containing the cross-validation Macro F1 and
Accuracy, plus the Test Macro F1 and Test Accuracy. The best K is
highlighted in bold. The tables are printed to the console and saved to
the `results/` folder as `.md` and `.csv` files.

### 7. Visualization

The project generates the following visualizations:

-   **F1-Score vs. K:** Shows how the Macro F1-Score changes for
    different K values.
-   **Confusion Matrix:** Visualizes correct and incorrect
    classifications.
-   **Decision Boundary:** Displays the classification regions for the
    two selected features.

## Technologies and Libraries

  Library        Purpose
  -------------- ------------------------------------------------------
  Pandas         Data loading and manipulation
  NumPy          Numerical operations
  Matplotlib     Data visualization
  Scikit-learn   KNN, preprocessing, cross-validation, and evaluation

## Installation

Install the required libraries:

``` bash
pip install pandas numpy matplotlib scikit-learn
```

## Project Structure

``` text
Iris-KNN-Classification/
│
├── data/
│   └── iris.csv
│
├── main.py
│
├── results/
│   ├── sepal_all_k.csv / .md
│   └── petal_all_k.csv / .md
│
└── README.md
```

## How to Run

1.  Clone or download the repository.
2.  Make sure the dataset is located at `data/iris.csv`.
3.  Install the required dependencies.
4.  Run the Python script:

``` bash
python main.py
```

## Results

The best K is selected using the **mean 5-Fold CV Macro F1-Score** on the
training set only. The Test columns are shown for information and are
**not** used to choose K. If several K values share the same CV F1, the
smallest K is chosen (the default `GridSearchCV` behavior).

### Sepal Features (Sepal Length, Sepal Width)

**Best K = 5** (CV Macro F1 = 0.7903)

| K | CV Macro F1 | CV Accuracy | Test Macro F1 | Test Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.7749 | 0.7750 | 0.6931 | 0.7000 |
| 2 | 0.7565 | 0.7667 | 0.5378 | 0.5667 |
| 3 | 0.7566 | 0.7583 | 0.6512 | 0.6667 |
| 4 | 0.7747 | 0.7833 | 0.7613 | 0.7667 |
| **5 (best)** | **0.7903** | **0.7917** | **0.6992** | **0.7000** |
| 6 | 0.7665 | 0.7750 | 0.6337 | 0.6667 |
| 7 | 0.7456 | 0.7500 | 0.7333 | 0.7333 |
| 8 | 0.7525 | 0.7583 | 0.6528 | 0.6667 |
| 9 | 0.7514 | 0.7583 | 0.6633 | 0.6667 |
| 10 | 0.7514 | 0.7583 | 0.7222 | 0.7333 |
| 11 | 0.7601 | 0.7667 | 0.7661 | 0.7667 |
| 12 | 0.7670 | 0.7750 | 0.7661 | 0.7667 |
| 13 | 0.7683 | 0.7750 | 0.6992 | 0.7000 |
| 14 | 0.7683 | 0.7750 | 0.6992 | 0.7000 |
| 15 | 0.7683 | 0.7750 | 0.6992 | 0.7000 |
| 16 | 0.7776 | 0.7833 | 0.6992 | 0.7000 |
| 17 | 0.7695 | 0.7750 | 0.7306 | 0.7333 |
| 18 | 0.7776 | 0.7833 | 0.7306 | 0.7333 |
| 19 | 0.7695 | 0.7750 | 0.7306 | 0.7333 |
| 20 | 0.7864 | 0.7917 | 0.7333 | 0.7333 |
| 21 | 0.7700 | 0.7750 | 0.7306 | 0.7333 |
| 22 | 0.7774 | 0.7833 | 0.5960 | 0.6000 |
| 23 | 0.7774 | 0.7833 | 0.7306 | 0.7333 |
| 24 | 0.7678 | 0.7750 | 0.7306 | 0.7333 |
| 25 | 0.7774 | 0.7833 | 0.7306 | 0.7333 |
| 26 | 0.7687 | 0.7750 | 0.6992 | 0.7000 |
| 27 | 0.7607 | 0.7667 | 0.7306 | 0.7333 |
| 28 | 0.7678 | 0.7750 | 0.7306 | 0.7333 |
| 29 | 0.7678 | 0.7750 | 0.7306 | 0.7333 |
| 30 | 0.7678 | 0.7750 | 0.6992 | 0.7000 |

### Petal Features (Petal Length, Petal Width)

**Best K = 4** (CV Macro F1 = 0.9665)

| K | CV Macro F1 | CV Accuracy | Test Macro F1 | Test Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9413 | 0.9417 | 0.9666 | 0.9667 |
| 2 | 0.9410 | 0.9417 | 0.9327 | 0.9333 |
| 3 | 0.9496 | 0.9500 | 0.9666 | 0.9667 |
| **4 (best)** | **0.9665** | **0.9667** | **1.0000** | **1.0000** |
| 5 | 0.9665 | 0.9667 | 0.9666 | 0.9667 |
| 6 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 7 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 8 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 9 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 10 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 11 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 12 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 13 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 14 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 15 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 16 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 17 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 18 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 19 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 20 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 21 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 22 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 23 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 24 | 0.9580 | 0.9583 | 0.9333 | 0.9333 |
| 25 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 26 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 27 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 28 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 29 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |
| 30 | 0.9665 | 0.9667 | 0.9333 | 0.9333 |

### Summary

| Feature Group | Best K | CV Macro F1 | Test Accuracy | Test Macro F1 |
|:---:|:---:|:---:|:---:|:---:|
| Sepal | 5 | 0.7903 | 0.7000 | 0.6992 |
| Petal | 4 | 0.9665 | 1.0000 | 1.0000 |

> **Note:** these numbers were generated with the standard Iris dataset
> (`random_state=42`). If your local `data/iris.csv` differs slightly,
> the values can change, so re-run the code to regenerate the tables
> in `results/`.

## Conclusion

This project demonstrates how feature selection can affect the
performance of a KNN classification model.

By evaluating Sepal and Petal features separately, the project
investigates which feature group provides better classification
performance.

Using GridSearchCV and Macro F1-Score helps identify an optimal K value
while maintaining a separate test set for final evaluation.

The results provide a comparison of the predictive performance of the
two feature groups.

------------------------------------------------------------------------

**Algorithm:** K-Nearest Neighbors (KNN)\
**Dataset:** Iris Flower Dataset\
**Optimization:** GridSearchCV\
**Evaluation Metric:** Macro F1-Score\
**Cross Validation:** Stratified 5-Fold
