# ============================================================
# Customer Churn Prediction and Behavioral Analysis
# E-Commerce Platform
# ============================================================

# ── 1. Imports ───────────────────────────────────────────────
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
)
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import shap


# ── 2. Load Data ─────────────────────────────────────────────
data = pd.read_excel("/content/data.xlsx", sheet_name="E Comm")

print(data.head(10))
print(data.info())
print(data.describe())
print("Shape:", data.shape)


# ── 3. Missing Value Analysis ─────────────────────────────────
missing_percent = data.isnull().mean() * 100
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

plt.figure(figsize=(6, 3))
sns.heatmap(data.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()


# ── 4. Skewness Check ────────────────────────────────────────
columns = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp",
    "OrderAmountHikeFromlastYear", "CouponUsed",
    "OrderCount", "DaySinceLastOrder", "CashbackAmount",
]
for col in columns:
    print(f"Skewness of {col}: {data[col].skew():.2f}")


# ── 5. Missing Value Imputation ───────────────────────────────
# Median imputation for highly skewed columns
cols_median = [
    "Tenure", "WarehouseToHome", "OrderAmountHikeFromlastYear",
    "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount",
]
for col in cols_median:
    data[col] = data[col].fillna(data[col].median())

# Mean imputation for near-symmetric column
data["HourSpendOnApp"] = data["HourSpendOnApp"].fillna(data["HourSpendOnApp"].mean())

# Verify no missing values remain
print(data.isnull().mean() * 100)


# ── 6. Duplicate Check ───────────────────────────────────────
num_duplicates = data.duplicated(subset=["CustomerID"]).sum()
print(f"Number of repeated customer IDs: {num_duplicates}")


# ── 7. EDA – Numerical Feature Distributions ─────────────────
num_cols = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "NumberOfAddress", "OrderAmountHikeFromlastYear",
    "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount",
]

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(data[col], kde=True, ax=axes[i], color="skyblue")
    axes[i].set_title(f"{col} (Skew = {data[col].skew():.2f})")
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()


# ── 8. EDA – Categorical Feature Distributions ───────────────
cat_cols = [
    "Churn", "PreferredLoginDevice", "CityTier", "PreferredPaymentMode",
    "Gender", "PreferedOrderCat", "SatisfactionScore",
    "MaritalStatus", "Complain",
]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    sns.countplot(
        data=data, x=col,
        order=data[col].value_counts().index,
        ax=axes[i], color="purple",
    )
    axes[i].set_title(f"{col} Distribution")
    axes[i].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()


# ── 9. Correlation Heatmap ───────────────────────────────────
numeric_data = data.select_dtypes(include=["int64", "float64"])
corr_matrix = numeric_data.corr()

plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ── 10. Scatter Plots – Feature Pairs ────────────────────────
feature_pairs = [
    ("CouponUsed", "OrderCount"),
    ("OrderCount", "DaySinceLastOrder"),
    ("CashbackAmount", "Tenure"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (x, y) in enumerate(feature_pairs):
    sns.scatterplot(data=data, x=x, y=y, ax=axes[i], alpha=0.6)
    axes[i].set_title(f"{x} vs {y}", fontsize=12)
    axes[i].grid(True)
plt.tight_layout()
plt.show()


# ── 11. Feature Engineering ───────────────────────────────────
data["CouponPerOrder"] = data["CouponUsed"] / data["OrderCount"].replace(0, np.nan)

sns.histplot(data["CouponPerOrder"], kde=True, color="skyblue")
plt.show()

missing_coupon = data["CouponPerOrder"].isnull().mean() * 100
print(f"Missing values in CouponPerOrder: {missing_coupon:.2f}%")


# ── 12. Churn vs Numerical Features ──────────────────────────
num_cols_fe = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "NumberOfAddress", "OrderAmountHikeFromlastYear",
    "CouponPerOrder", "OrderCount", "DaySinceLastOrder", "CashbackAmount",
]

n_cols_plot = 3
n_rows_plot = math.ceil(len(num_cols_fe) / n_cols_plot)
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(n_cols_plot * 6, n_rows_plot * 4))
axes = axes.flatten()
for i, col in enumerate(num_cols_fe):
    sns.histplot(data=data, x=col, hue="Churn", ax=axes[i], kde=True)
    axes[i].set_title(f"{col} vs Churn")
    axes[i].tick_params(axis="x", rotation=45)
for j in range(len(num_cols_fe), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()


# ── 13. Churn vs Categorical Features ────────────────────────
cat_cols_churn = [
    "PreferredLoginDevice", "CityTier", "PreferredPaymentMode",
    "Gender", "PreferedOrderCat", "SatisfactionScore",
    "MaritalStatus", "Complain",
]

n_rows_cat = (len(cat_cols_churn) + 2) // 3
fig, axes = plt.subplots(nrows=n_rows_cat, ncols=3, figsize=(18, n_rows_cat * 5))
axes = axes.flatten()
for i, col in enumerate(cat_cols_churn):
    sns.countplot(data=data, x=col, hue="Churn", ax=axes[i])
    axes[i].set_title(f"{col} vs Churn")
    axes[i].tick_params(axis="x", rotation=45)
for j in range(len(cat_cols_churn), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()


# ── 14. Preprocessing ────────────────────────────────────────
X = data.drop(columns=["Churn", "CustomerID", "CouponUsed"])
y = data["Churn"]

X_encoded = pd.get_dummies(X, columns=cat_cols_churn, drop_first=True)

scaler = StandardScaler()
X_encoded[num_cols_fe] = scaler.fit_transform(X_encoded[num_cols_fe])
X = X_encoded


# ── 15. Train / Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ── 16. Baseline Model Training ──────────────────────────────
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)


# ── 17. Model Evaluation ─────────────────────────────────────
models = {
    "Logistic Regression": y_pred_log,
    "Decision Tree": y_pred_tree,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb,
}
probs = {
    "Logistic Regression": log_model.predict_proba(X_test)[:, 1],
    "Decision Tree": tree_model.predict_proba(X_test)[:, 1],
    "Random Forest": rf_model.predict_proba(X_test)[:, 1],
    "XGBoost": xgb_model.predict_proba(X_test)[:, 1],
}

# Confusion matrices
plt.figure(figsize=(14, 10))
for i, (name, y_pred) in enumerate(models.items(), 1):
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(2, 2, i)
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC curves
plt.figure(figsize=(10, 6))
for name, prob in probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# Metrics summary
metrics_df = []
for name, y_pred in models.items():
    metrics_df.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, probs[name]),
    })
metrics_df = pd.DataFrame(metrics_df)
print(metrics_df)


# ── 18. XGBoost Hyperparameter Tuning ────────────────────────
xgb_base = XGBClassifier()

param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [5, 8, 14, 17, 20, 25],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.3, 0.4, 0.6, 0.7],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3],
    "reg_lambda": [0.01, 0.1, 1, 1.5, 3],
}

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=40,
    scoring="f1",
    cv=5,
    random_state=42,
)
random_search.fit(X_train, y_train)

best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Tuned F1 Score: {f1_score(y_test, y_pred)}")


# ── 19. Final Model Evaluation ───────────────────────────────
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

y_proba = best_xgb.predict_proba(X_test)[:, 1]
print(f"Accuracy:   {accuracy_score(y_test, y_pred)}")
print(f"Precision:  {precision_score(y_test, y_pred)}")
print(f"Recall:     {recall_score(y_test, y_pred)}")
print(f"F1-Score:   {f1_score(y_test, y_pred)}")
print(f"ROC-AUC:    {roc_auc_score(y_test, y_proba)}")


# ── 20. Feature Importance & SHAP ────────────────────────────
plot_importance(best_xgb, max_num_features=10)
plt.title("Top Features - XGBoost")
plt.show()

explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)
shap.plots.waterfall(shap_values[3])
shap.plots.bar(shap_values)