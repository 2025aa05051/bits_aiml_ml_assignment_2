import streamlit as st
import pandas as pd
import joblib
#from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
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

# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 - Bank Marketing",
    layout="wide"
)

st.title("ML Assignment 2 – Bank Marketing Classification")
st.write(
    """
    This Streamlit application demonstrates multiple classification models
    trained on the Bank Marketing dataset.
    Upload test data, select a model, and view evaluation results.
    """
)

# -------------------------------------------------
# Model registry
# -------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "model/artifacts/logistic_regression.pkl",
    "Decision Tree": "model/artifacts/decision_tree.pkl",
    "KNN": "model/artifacts/knn.pkl",
    "Naive Bayes": "model/artifacts/naive_bayes.pkl",
    "Random Forest": "model/artifacts/random_forest.pkl",
    "XGBoost": "model/artifacts/xgboost.pkl",
}

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("Controls")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (must contain target column 'y')",
    type=["csv"]
)

# -------------------------------------------------
# Helper: compute metrics
# -------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

# -------------------------------------------------
# Main logic
# -------------------------------------------------
if uploaded_file is None:
    st.info(
        "Please upload the test CSV file.\n\n"
        "Tip: Use the `data/test.csv` generated during model training."
    )
    st.stop()

# Load test data
df = pd.read_csv(uploaded_file, sep=";")

strngs  = ''
for a in df.columns :
    strngs += a + ' ,'
if "y" not in df.columns:
    st.error("Uploaded CSV must include the target column 'y'." \
    "given columns where : " +strngs)
    st.stop()

y_true_raw = df["y"]

y_true = df["y"]

# Convert y to 0/1 if it is "yes"/"no"
if y_true_raw.dtype == object:
    y_true = y_true_raw.map({"yes": 1, "no": 0})
else:
    y_true = y_true_raw

# Safety check
if y_true.isna().any():
    st.error("Target column 'y' has values other than yes/no or 0/1. Please check the uploaded file.")
    st.stop()

X = df.drop(columns=["y"])

# Load selected model
model_path = MODEL_FILES[selected_model]
model = joblib.load(model_path)

# Predictions
y_pred = model.predict(X)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X)[:, 1]
else:
    # fallback for models without predict_proba
    scores = model.decision_function(X)
    y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

metrics = compute_metrics(y_true, y_pred, y_proba)

# # -------------------------------------------------
# # Display results
# # -------------------------------------------------
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Evaluation Metrics")
#     st.dataframe(pd.DataFrame([metrics]))

# with col2:
#     st.subheader("Confusion Matrix")
#     cm = confusion_matrix(y_true, y_pred)
#     st.write(cm)

# st.subheader("Classification Report")
# st.text(classification_report(y_true, y_pred))


# ---- Display results (better UI) ----
st.subheader(f"Results (Evaluation Metrics ) for: {selected_model}")

# Metrics as KPI cards
m1, m2, m3 = st.columns(3)
m4, m5, m6 = st.columns(3)

m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
m2.metric("AUC", f"{metrics['AUC']:.4f}")
m3.metric("Precision", f"{metrics['Precision']:.4f}")

m4.metric("Recall", f"{metrics['Recall']:.4f}")
m5.metric("F1 Score", f"{metrics['F1']:.4f}")
m6.metric("MCC", f"{metrics['MCC']:.4f}")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)  # default colormap; no seaborn needed

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (no)", "1 (yes)"])
    ax.set_yticklabels(["0 (no)", "1 (yes)"])

    # Annotate values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    st.pyplot(fig)

with right:
    st.subheader("Classification Report")

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    # Make it prettier
    report_df = report_df.round(4)
    st.dataframe(report_df, use_container_width=True)
