# Breast Cancer Classification

A machine learning classification project that predicts whether a breast tumor is **malignant or benign** using numerical tumor features from a breast cancer dataset.

The project implements a complete machine learning workflow, including exploratory data analysis, preprocessing, model comparison, evaluation, and hyperparameter tuning.

## Workflow

The project includes:

- Data loading and exploration
- Missing-value handling
- Exploratory Data Analysis (EDA)
- Diagnosis distribution visualization
- Correlation analysis
- Train/test splitting with stratification
- Feature standardization using `StandardScaler`
- Logistic Regression
- Random Forest Classification
- Model performance comparison
- Classification reports
- Confusion matrices
- ROC Curve and AUC evaluation
- Random Forest hyperparameter tuning using `GridSearchCV`
- 5-fold cross-validation

## Models

### Logistic Regression

Used as a classification model after standardizing the numerical features.

### Random Forest

Used as an ensemble classification model and further optimized through hyperparameter tuning.

The Grid Search evaluates different values for:

- Number of estimators
- Maximum tree depth
- Minimum samples required to split a node

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- AUC

The best-performing model is selected based on test accuracy before additional Random Forest tuning is performed.

## Technologies

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The project uses `Breast_cancer_data.csv`, containing numerical tumor measurements and a diagnosis target:

- `0` — Malignant
- `1` — Benign

## Academic Context

This project was developed as part of university coursework to apply machine learning concepts to a classification problem using healthcare-related data.

> This project is intended for educational purposes only and is not a medical diagnostic tool.
