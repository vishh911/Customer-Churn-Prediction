# Customer Churn Prediction and Behavioral Analysis

A Machine Learning project focused on predicting customer churn for an E-Commerce platform using classification models and behavioral analysis techniques.

This project performs complete data preprocessing, exploratory data analysis (EDA), feature engineering, model building, hyperparameter tuning, and explainable AI using SHAP values.

---

## Project Objective

The objective of this project is to:

- Predict whether a customer is likely to churn
- Analyze behavioral patterns contributing to churn
- Compare multiple machine learning models
- Improve prediction performance using XGBoost tuning
- Interpret model predictions using SHAP explainability

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP

---

## Models Implemented

The following classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier

---

## Workflow

### 1. Data Loading
- Imported customer data from Excel dataset

### 2. Data Preprocessing
- Missing value analysis
- Median and mean imputation
- Duplicate customer detection

### 3. Exploratory Data Analysis
- Distribution plots
- Count plots
- Correlation heatmap
- Scatter plots

### 4. Feature Engineering
Created a new feature:
- `CouponPerOrder`

### 5. Encoding & Scaling
- One-hot encoding for categorical variables
- Standardization using `StandardScaler`

### 6. Model Training
- Train-test split
- Baseline model training

### 7. Model Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Additional evaluation:
- Confusion Matrix
- ROC Curve

### 8. Hyperparameter Tuning
Performed Randomized Search Cross Validation on XGBoost.

### 9. Explainable AI
Used SHAP for:
- Feature importance
- Global interpretability
- Local prediction explanations

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data.xlsx
├── customer_churn.py
├── README.md
└── documentation/
    └── Project_Documentation.md