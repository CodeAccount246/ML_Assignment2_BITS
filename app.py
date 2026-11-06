import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Netflix Content Classification App",
    page_icon="🍿",
    layout="wide"
)

st.title("🍿 Netflix Content Classification Model Evaluator")
st.markdown("### Interactive Model Evaluation Dashboard | Predicting Movie vs TV Show")
st.markdown("---")

st.sidebar.header("🕹️ Configuration Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Data (CSV)", 
    type=["csv"],
    help="Upload the raw netflix_titles.csv or the processed test_data.csv."
)

model_choice = st.sidebar.selectbox(
    "Select ML Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    )
)

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

@st.cache_resource
def load_scaler_and_model(name):
    scaler = joblib.load("model/scaler.joblib")
    model = joblib.load(model_files[name])
    return scaler, model

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    
    # --- DYNAMIC FEATURE ENGINEERING FOR RAW DATA ---
    raw_required_columns = ['type', 'date_added', 'description', 'cast', 'listed_in', 'release_year']
    
    # Check if the uploaded dataset is the raw netflix_titles.csv by seeing if it has these textual columns
    if all(col in df_test.columns for col in raw_required_columns):
        st.sidebar.info("Raw dataset detected. Applying feature engineering...")
        
        # 1. Target column
        df_test['target'] = (df_test['type'] == 'TV Show').astype(int)
        
        # 2. Extract Year
        df_test['added_year'] = pd.to_datetime(df_test['date_added'].str.strip(), format='%B %d, %Y', errors='coerce').dt.year
        
        # 3. Calculate text lengths & list counts
        df_test['desc_len'] = df_test['description'].fillna('').apply(len)
        df_test['cast_count'] = df_test['cast'].fillna('').apply(lambda x: len(str(x).split(',')) if pd.notnull(x) and x != '' else 0)
        df_test['listed_in_count'] = df_test['listed_in'].fillna('').apply(lambda x: len(str(x).split(',')) if pd.notnull(x) and x != '' else 0)
        
        # 4. Filter missing data to mirror training
        features_to_check = ['release_year', 'added_year', 'desc_len', 'cast_count', 'listed_in_count', 'target']
        df_test = df_test.dropna(subset=features_to_check)
        
    st.sidebar.success(f"Loaded {df_test.shape[0]} valid rows and {df_test.shape[1]} columns.")
    
    scaler, model = load_scaler_and_model(model_choice)
    
    # Get expected features from the scaler
    expected_features = getattr(scaler, "feature_names_in_", None)
    
    if expected_features is not None:
        missing_cols = [col for col in expected_features if col not in df_test.columns]
        if missing_cols:
            st.error(f"Uploaded CSV is missing expected features: {missing_cols}. Ensure you uploaded the correct Netflix CSV.")
            st.stop()
        
        # Ensure correct column order
        X_test = df_test[expected_features]
    else:
        # Fallback if no feature names are saved
        if "target" in df_test.columns:
            X_test = df_test.drop(columns=["target"])
        else:
            X_test = df_test.iloc[:, :-1]
            
    # Set target
    if "target" in df_test.columns:
        y_test = df_test["target"]
    else:
        y_test = df_test.iloc[:, -1]

    if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        # scaler.transform returns a numpy array, perfectly fine for models trained on scaled arrays
        X_eval = scaler.transform(X_test)
    else:
        # Decision Tree & Naive Bayes were fitted on unscaled DataFrames, preserve feature names
        X_eval = X_test

    try:
        y_pred = model.predict(X_eval)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_eval)[:, 1]
        else:
            y_proba = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader(f"📌 Evaluation Metrics: {model_choice}")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")
        
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax,
                        xticklabels=["Movie (0)", "TV Show (1)"],
                        yticklabels=["Movie (0)", "TV Show (1)"])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            st.pyplot(fig)

        with col_right:
            st.subheader("📋 Classification Report")
            report = classification_report(
                y_test, y_pred, 
                target_names=["Movie (0)", "TV Show (1)"],
                zero_division=0
            )
            st.code(report, language="text")
    except Exception as e:
        st.error(f"Prediction error: {e}")

else:
    st.info("👈 Upload `test_data.csv` or `netflix_titles.csv` using the sidebar to view evaluation metrics, confusion matrix, and classification report.")
