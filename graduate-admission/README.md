# Graduate Admission Prediction

A supervised machine learning project that uses **Linear Regression** to predict a student's chance of graduate admission based on selected academic features.

## Features Used

The model uses:

- GRE Score
- TOEFL Score
- CGPA
- Research Experience

The target variable is the student's **Chance of Admit**.

## Workflow

The project includes:

- Dataset loading and preparation
- Feature and target selection
- Train/test splitting
- Feature standardization using `StandardScaler`
- Linear Regression training
- Prediction on test data
- Comparison of actual and predicted values
- Visualization of CGPA against admission chance

## Model Evaluation

The regression model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Files

- `grad_admission_regression.py` — preprocessing, model training, evaluation, and visualization
- `Admission_Predict.csv` — dataset used by the project
