#%% md
# # **MACHINE LEARNING**
# 
# # **ASSEIGNMENT NO 02**
# 
# **Student Name:** KALE CHAITANYA PRASAD
# 
# **Student ID:** 2025AA05377
# 
# **Date:** 10.02.2026
# 
# 
# # Student Academic Placement Performance Dataset
#%%
import streamlit as st
import pandas as pd
import numpy as np
import scikit-learn as sk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
#%%
st.set_page_config(
    page_title="Student Placement Prediction",
    layout="centered"
)

st.title("🎓 Student Placement & Salary Range Prediction")
st.write("Classification-based prediction (No Regression Used)")

#%%
@st.cache_data
def load_data():
    df = pd.read_csv("student_academic_placement_performance_dataset.csv")
    if "sl_no" in df.columns:
        df.drop(columns=["sl_no"], inplace=True)
    return df

df = load_data()
#%%
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

#%%
X1 = df.drop(
    columns=["placement_status", "salary_package_lpa"],
    errors="ignore"
)

y1 = df["placement_status"]

scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1_scaled,
    y1,
    test_size=0.2,
    random_state=42,
    stratify=y1
)

placement_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
placement_model.fit(X1_train, y1_train)
#%%
## SALARY RANGE CLASSIFICATION (PLACED ONLY)
placed_df = df[df["placement_status"] == 1].copy()

# Define salary bins (LPA)
salary_bins = [0, 3, 6, 10, np.inf]

placed_df["salary_range"] = pd.cut(
    placed_df["salary_package_lpa"],
    bins=salary_bins
)

# Re-encode salary classes to 0,1,2,...
salary_encoder = LabelEncoder()
placed_df["salary_class"] = salary_encoder.fit_transform(
    placed_df["salary_range"]
)

X2 = placed_df.drop(
    columns=[
        "placement_status",
        "salary_package_lpa",
        "salary_range",
        "salary_class"
    ]
)

y2 = placed_df["salary_class"]

scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_scaled,
    y2,
    test_size=0.2,
    random_state=42,
    stratify=y2
)

salary_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
salary_model.fit(X2_train, y2_train)
#%%
## MODEL DEFINITIONS
binary_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
}
#%%
multiclass_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y2)),
        eval_metric="mlogloss",
        random_state=42
    )
}
#%%
## EVALUATION FUNCTION
def evaluate_models(models, X_train, X_test, y_train, y_test, task_type):
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "binary":
            y_prob = model.predict_proba(X_test)[:, 1]

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "MCC": matthews_corrcoef(y_test, y_pred)
            })

        else:  # multiclass
            y_prob = model.predict_proba(X_test)

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(
                    y_test, y_prob, multi_class="ovr"
                ),
                "Precision": precision_score(
                    y_test, y_pred, average="weighted"
                ),
                "Recall": recall_score(
                    y_test, y_pred, average="weighted"
                ),
                "F1 Score": f1_score(
                    y_test, y_pred, average="weighted"
                ),
                "MCC": matthews_corrcoef(y_test, y_pred)
            })

    return pd.DataFrame(results)
#%%
placement_results = evaluate_models(
    binary_models,
    X1_train, X1_test,
    y1_train, y1_test,
    task_type="binary"
)

salary_results = evaluate_models(
    multiclass_models,
    X2_train, X2_test,
    y2_train, y2_test,
    task_type="multiclass"
)
#%%
print("\nPLACEMENT STATUS PREDICTION RESULTS\n")
print(placement_results.sort_values("MCC", ascending=False))

print("\nSALARY RANGE PREDICTION RESULTS\n")
print(salary_results.sort_values("MCC", ascending=False))
#%%
st.header("📥 Enter Student Details")

user_input = []
for col in X1.columns:
    val = st.number_input(
        label=f"{col}",
        value=0.0,
        step=1.0
    )
    user_input.append(val)

user_input = np.array(user_input).reshape(1, -1)
user_input_scaled = scaler1.transform(user_input)
#%%
if st.button("🔍 Predict"):
    placement_pred = placement_model.predict(user_input_scaled)[0]
    placement_prob = placement_model.predict_proba(user_input_scaled)[0][1]

    if placement_pred == 1:
        st.success(f"✅ PLACED (Confidence: {placement_prob:.2f})")

        salary_scaled = scaler2.transform(user_input)
        salary_class = salary_model.predict(salary_scaled)[0]
        salary_range = salary_encoder.inverse_transform([salary_class])[0]

        st.info(f"💰 Predicted Salary Range (LPA): {salary_range}")

    else:
        st.error(f"❌ NOT PLACED (Confidence: {1 - placement_prob:.2f})")
