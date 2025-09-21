# Linear and Non-Linear SVM for Classification and Regression Problems

## Table of Contents

1.  [Overview](#overview)
2.  [Setup](#setup)
3.  [Data](#data)
4.  [Classification](#classification)
    *   [Data](#classification-data)
    *   [Data Preprocessing](#classification-data-preprocessing)
    *   [Model Implementation](#classification-model-implementation)
5.  [Regression](#regression)
    *   [Data](#regression-data)
    *   [Data Preprocessing](#regression-data-preprocessing)
    *   [Model Implementation](#regression-model-implementation)
6.  [Comparison](#comparison)
    *   [Classification: Linear vs Non-Linear SVM](#classification-linear-vs-non-linear-svm)
    *   [Regression: Linear vs Non-Linear SVM](#regression-linear-vs-non-linear-svm)
7.  [Conclusion](#conclusion)
8.  [License](#license)

## Overview

This project explores the application of Support Vector Machines (SVM) with both linear and non-linear kernels (specifically, the Radial Basis Function or RBF kernel) to solve classification and regression problems. The project includes data loading, preprocessing, feature analysis, model training, evaluation, and visualization of results. The goal is to demonstrate the strengths and weaknesses of each approach on different types of datasets.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install the required packages:**

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
    ```
3.  **Jupyter Notebook:**

    The code is implemented in a Jupyter Notebook (`linear_and_non_linear_SVM_for_classification_and_regression_problems.ipynb`). Ensure you have Jupyter installed or use Google Colab to run the notebook.

## Data

This project utilizes two distinct datasets, one for classification and another for regression tasks. Details on each are provided in their respective sections below.

## Classification

### Data

Dataset before cleaning 

<img width="1758" height="164" alt="image" src="https://github.com/user-attachments/assets/f250636e-04c6-4c5e-9b57-ee13f5ddc298" />

Dataset after cleaning 

<img width="600" height="100" alt="image" src="https://github.com/user-attachments/assets/80be235c-8abd-4986-91f1-a265b26f31d7" />

The classification task uses the "rock\_classification\_dataset.csv" dataset. This dataset is used to classify different types of rocks based on various features.

### Data Preprocessing

1.  **Loading the Data:** The dataset is loaded using pandas.
2.  **Exploratory Data Analysis (EDA):**
    *   Displaying the dataset.
    *   Displaying the dataset shape (number of rows and columns).
    *   Displaying columns line by line.
    *   Displaying data type of each column.
    *   Displaying the number of missing values in each column.
    *   Summary statistics for numerical columns.
    *   Number of unique values in each column.
    *   Checking for duplicate rows.
    *   Computing the correlation matrix. This helps understand the relationships between features.
3.  **Variance Inflation Factor (VIF):** VIF is calculated to detect multicollinearity among features. High VIF scores indicate that some features are highly correlated, which can affect model performance.
4.  **Outlier Removal:** Outliers are removed using the IQR method within each class. This helps to reduce the impact of extreme values on the model.
5.  **Data Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique) is used to balance the class distribution. This ensures that the model is not biased towards the majority class.

# Data Imbalance Comparison

<table>
  <tr>
    <td>Data imbalance before</td>
    <td>Data imbalance after</td>
  </tr>
  <tr>
    <td><img width="300" height="300" alt="before" src="https://github.com/user-attachments/assets/785a3172-2560-4127-9440-90ace3d37d88" /></td>
    <td><img width="300" height="300" alt="after" src="https://github.com/user-attachments/assets/8d0576ec-d77d-4acf-b058-8ce1ac7c7ddf" /></td>
  </tr>
</table>

6.  **Feature Scaling:** StandardScaler is used to standardize the numerical features. Scaling ensures that all features contribute equally to the model and prevents features with larger values from dominating.
7.  **Splitting the Data:** The dataset is split into training and testing sets (80-20 ratio). This allows us to train the model on one set of data and evaluate its performance on unseen data.

### Model Implementation

1.  **Linear SVM:**
    *   GridSearchCV is used to find the best hyperparameters for the Linear SVM model. This automates the process of hyperparameter tuning by exhaustively searching through a specified subset of the hyperparameter space.
    *   The model is trained using the best estimator from GridSearchCV.
2.  **Non-Linear SVM (RBF Kernel):**
    *   GridSearchCV is used to find the best hyperparameters for the RBF SVM model.
    *   The model is trained using the best estimator from GridSearchCV.
3.  **Model Evaluation:**
    *   Accuracy score
    *   Classification report (precision, recall, F1-score): Provides a detailed analysis of the model's performance for each class.
    *   Confusion matrix: Visualizes the performance of the classification model by showing the counts of true positive, true negative, false positive, and false negative predictions.
    *   ROC AUC score

Linear SVM Evaluation:

Accuracy: 0.9592021758839528

Classification Report:

          precision    recall  f1-score   support

       1       0.98      0.98      0.98       595
       2       0.97      0.98      0.97       627
       3       1.00      1.00      1.00       600
       4       0.97      0.97      0.97       677
       5       0.99      0.99      0.99       635
       6       0.91      0.90      0.90       662
       7       0.90      0.91      0.90       616

accuracy                           0.96      4412

macro avg 0.96 0.96 0.96 4412
weighted avg 0.96 0.96 0.96 4412


confusion matrix linear

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/c9b764cc-336f-4695-83c8-7f7d75f7560e" />

RBF SVM Evaluation:

Accuracy: 0.9637352674524026

Classification Report:

          precision    recall  f1-score   support

       1       0.98      0.98      0.98       595
       2       0.99      0.98      0.98       627
       3       1.00      1.00      1.00       600
       4       0.98      0.99      0.98       677
       5       1.00      0.99      0.99       635
       6       0.91      0.91      0.91       662
       7       0.90      0.91      0.90       616

accuracy                           0.96      4412

macro avg 0.96 0.96 0.96 4412
weighted avg 0.96 0.96 0.96 4412


confusion matrix non linear

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/448e79b0-f3ec-45c3-8b68-3b695e17bc44" />

4.  **Decision Boundary Visualization:**
    *   PCA is used to reduce the features to 2D for visualization.
    *   The decision boundary is plotted using a mesh grid. This helps to visualize how the model separates the different classes.

linear decision boundary

<img width="330" height="270" alt="image" src="https://github.com/user-attachments/assets/e2a88e16-fbcf-4523-9eb7-119563add7ac" />

non linear decision boundary

<img width="420" height="250" alt="image" src="https://github.com/user-attachments/assets/88ca5092-94ad-4c4a-9c61-a7356ed04356" />

5.  **Cross-Validation**
    *   5-fold cross-validation is performed to assess the model's performance. This provides a more robust estimate of the model's performance by averaging the results across multiple splits of the data.
    *   The cross-validation scores are visualized.
    *   Comparison of accuracy before and after k-fold cross-validation.

## Regression

### Data

The regression task uses the "theme\_park\_visitor\_count\_dataset.csv" dataset. This dataset is used to predict the number of theme park visitors based on various features.

Dataset for regression before

<img width="1391" height="164" alt="image" src="https://github.com/user-attachments/assets/cbcbb0ac-4ec2-4a8b-964a-f64a23f9e76c" />

Dataset for regression after

<img width="1031" height="166" alt="image" src="https://github.com/user-attachments/assets/3ea01762-1774-489f-8831-16fd52446ba9" />

### Data Preprocessing

1.  **Loading the Data:** The dataset is loaded using pandas.
2.  **Exploratory Data Analysis (EDA):**
    *   Displaying the dataset
    *   Displaying the dataset shape (number of rows and columns)
    *   Displaying columns line by line
    *   Displaying data type of each column
    *   Displaying the number of missing values in each column
    *   Summary statistics for numerical columns
    *   Number of unique values in each column
    *   Checking for duplicate rows
    *   Computing the correlation matrix
3.  **Skewness Handling:** Log transformation is applied to reduce skewness in numerical features. Skewness can negatively impact the performance of some machine learning algorithms.
4.  **Outlier Removal:** Outliers are removed using the IQR method.
5.  **Feature Selection**
    *   Dropping holiday column
    *   Variance Inflation Factor (VIF):
    *   Correlation
6.  **Feature Scaling:** StandardScaler is used to standardize the numerical features.
7.  **Splitting the Data:** The dataset is split into training and testing sets (80-20 ratio).

### Model Implementation

1.  **Linear SVM Regression:**
    *   GridSearchCV is used to find the best hyperparameters for the Linear SVM Regression model.
    *   The model is trained using the best estimator from GridSearchCV.
2.  **Non-Linear SVM Regression (RBF Kernel):**
    *   RandomizedSearchCV is used to find the best hyperparameters for the RBF SVM Regression model. RandomizedSearchCV is used instead of GridSearchCV to reduce computational cost, especially when the hyperparameter space is large.
    *   The model is trained using the best estimator from RandomizedSearchCV.
3.  **Model Evaluation:**
    *   Mean Squared Error (MSE)
    *   Root Mean Squared Error (RMSE)
    *   Mean Absolute Error (MAE)
    *   R² score

Linear SVM Regression MSE: 6797.37
Linear SVM Regression RMSE: 82.45
Linear SVM Regression MAE: 59.30
Linear SVM Regression R²: 0.71

RBF SVM Regression MSE: 117.52
RBF SVM Regression RMSE: 10.84
RBF SVM Regression MAE: 4.66
RBF SVM Regression R²: 1.00

linear and non linear prediction

<img width="600" height="250" alt="image" src="https://github.com/user-attachments/assets/9800f61f-63d5-4367-92ce-541996530b9a" />

4.  **Cross-Validation**
    *   5-fold cross-validation is performed to assess the model's performance.

## Comparison

### Classification: Linear vs Non-Linear SVM

In the classification task, both Linear SVM and Non-Linear SVM (RBF Kernel) models were evaluated. The Linear SVM achieved good performance with an accuracy of 0.959, while the Non-Linear SVM (RBF Kernel) had a slightly higher accuracy of 0.964.  The choice between these models depends on the underlying structure of the data. Linear SVMs are preferred when the data is linearly separable or when computational resources are limited. Non-linear SVMs are better suited for complex, non-linear datasets.

### Regression: Linear vs Non-Linear SVM

In the regression task, both Linear SVM and Non-Linear SVM (RBF Kernel) models were evaluated. The Non-Linear SVM (RBF Kernel) significantly outperformed the Linear SVM, achieving an R² score of 1.00 compared to 0.71 for the Linear SVM. This indicates that the relationship between the features and the target variable is highly non-linear in this dataset.

## Conclusion

This project demonstrates the application of both Linear and RBF SVMs for classification and regression tasks. The choice of kernel depends on the specific dataset and problem. Linear SVMs are faster to train and are suitable for linearly separable data or when there are a large number of features. Non-linear kernels, like RBF, are more computationally intensive but can capture complex patterns in the data. For the rock classification dataset, both Linear and Non-Linear SVM performed well, indicating a potentially linearly separable problem. However, for the theme park visitor count dataset, the Non-Linear SVM significantly outperformed the Linear SVM, suggesting a non-linear relationship between the features and the target variable.

## License

[Specify the license under which the project is released, e.g., MIT License]
