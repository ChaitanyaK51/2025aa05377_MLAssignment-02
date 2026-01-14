#%%
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="ML Assignment 02", layout="centered")
st.title("🎓 Student Placement Prediction App")
st.write("Classification-based ML Models (No Regression Used)")

# --------------------------------------------------
# DATASET UPLOAD (FEATURE a)
# --------------------------------------------------
st.header("📂 Upload Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload Student Placement Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

if "sl_no" in df.columns:
    df.drop(columns=["sl_no"], inplace=True)

# --------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# --------------------------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# --------------------------------------------------
# TARGET SELECTION (PLACEMENT STATUS)
# --------------------------------------------------
st.header("🎯 Target: Placement Status")

X = df.drop(columns=["placement_status", "salary_package_lpa"], errors="ignore")
y = df["placement_status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# MODEL SELECTION DROPDOWN (FEATURE b)
# --------------------------------------------------
st.header("🧠 Select Classification Model")

model_name = st.selectbox(
    "Choose Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)"
    )
)

# Model mapping
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)

elif model_name == "kNN":
    model = KNeighborsClassifier(n_neighbors=5)

elif model_name == "Naive Bayes":
    model = GaussianNB()

elif model_name == "Random Forest (Ensemble)":
    model = RandomForestClassifier(n_estimators=200, random_state=42)

else:  # XGBoost
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# EVALUATION METRICS DISPLAY (FEATURE c)
# --------------------------------------------------
st.header("📊 Evaluation Metrics")

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ],
    "Value": [
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ]
})

st.table(metrics_df.style.format({"Value": "{:.3f}"}))

# --------------------------------------------------
# CONFUSION MATRIX & CLASSIFICATION REPORT (FEATURE d)
# --------------------------------------------------
st.header("📉 Model Performance Analysis")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
st.pyplot(fig)

st.subheader("📄 Classification Report")
st.text(classification_report(y_test, y_pred))
