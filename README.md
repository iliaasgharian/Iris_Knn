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

## Expected Results

After execution, the program identifies the optimal K value for both
feature groups.

The final comparison includes:

  Feature Group   Optimal K                  Test Accuracy   Test Macro F1
  --------------- -------------------------- --------------- ---------------
  Sepal           Automatically determined   Calculated      Calculated
  Petal           Automatically determined   Calculated      Calculated

*The actual values are generated when the program is executed.*

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
