import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# 1. Load Dataset & Feature Engineering
df = pd.read_csv("netflix_titles.csv")
df['target'] = (df['type'] == 'TV Show').astype(int)
df['added_year'] = pd.to_datetime(df['date_added'].str.strip(), format='%B %d, %Y', errors='coerce').dt.year
df['desc_len'] = df['description'].fillna('').apply(len)
df['cast_count'] = df['cast'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
df['listed_in_count'] = df['listed_in'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)

features = ['release_year', 'added_year', 'desc_len', 'cast_count', 'listed_in_count']
ml_df = df[features + ['target']].dropna()

X = ml_df[features]
y = ml_df['target']

# 2. Train-Test Split (80/20 Stratified Split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs("model", exist_ok=True)

test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv("test_data.csv", index=False)
print("Saved 'test_data.csv' to current working directory.")

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/scaler.joblib")

# 4. Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# 5. Train Models & Export
metrics_list = []
for name, model in models.items():
    if name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    file_name = f"model/{name.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, file_name)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics_list.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })

results_df = pd.DataFrame(metrics_list)
print("\n--- Model Evaluation Summary Table ---")
print(results_df.to_string(index=False))
