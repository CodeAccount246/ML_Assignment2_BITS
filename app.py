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
    page_icon="🎬",
    layout="wide"
)

st.title("Netflix Content Classification Model Evaluator")
st.markdown("### Interactive Model Evaluation Dashboard | Predicting Movie vs TV Show")
