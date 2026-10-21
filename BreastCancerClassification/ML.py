# Machine Learning Approach to Predict Breast Cancer

# 1) Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 2) Load the dataset
df = pd.read_csv("Breast_cancer_data.csv")

# Display basic dataset information
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

# 3) Handle Missing Values
# Replace missing numeric values with column means
df = df.fillna(df.mean(numeric_only=True))

# 4) Exploratory Data Analysis (EDA)
# Plot diagnosis distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["diagnosis"])
plt.title("Diagnosis Count (0 = Malignant, 1 = Benign)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# 5) Split Dataset into Features and Target
# Separate features (X) from target variable (y)
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Feature Scaling
# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7) Train Models (Logistic Regression + Random Forest)
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}
trained_models = {}

# Train each model and evaluate accuracy
for name, model in models.items():
    print(f"\n===== Training {name} =====")
    model.fit(X_train, y_train)
    trained_models[name] = model  # Save trained model

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# 8) Compare Model Performance
print("\n=== Model Comparison ===")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# Select the best-performing model
best_model_name = max(results, key=results.get)
print("\nBest Model is:", best_model_name)

best_model = trained_models[best_model_name]

# 9) Confusion Matrix + ROC Curve for Best Model
print(f"\n===== Evaluation for Best Model: {best_model_name} =====")
best_y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{best_model_name} - Confusion Matrix")
plt.show()

# Compute probabilities for ROC curve
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test)[:, 1]
else:
    try:
        y_scores = best_model.decision_function(X_test)
        y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    except Exception:
        y_prob = None

    # Plot ROC Curve if probability scores exist
if y_prob is not None:
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    print(f"AUC Score for {best_model_name}: {auc_score:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {best_model_name}")
    plt.legend()
    plt.show()

# 10) GridSearchCV for Random Forest (Hyperparameter Tuning)
print("\n===== Running Grid Search for Random Forest =====")
# 10) Hyperparameter Tuning (Random Forest with GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

# Perform grid search with 5-fold CV
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Display best parameters found
print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# Evaluate tuned Random Forest
best_rf = grid.best_estimator_
y_pred_grid = best_rf.predict(X_test)

print("\n===== Tuned Random Forest Evaluation =====")
print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print(classification_report(y_test, y_pred_grid))

# Confusion Matrix for tuned model
cm = confusion_matrix(y_test, y_pred_grid)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title("Tuned Random Forest - Confusion Matrix")
plt.show()