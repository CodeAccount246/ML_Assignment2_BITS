# Netflix Content Classification & Evaluation App

## a. Problem Statement
The objective of this project is to build, evaluate, and benchmark five distinct machine learning classification algorithms to predict whether a Netflix title is a **Movie** (`0`) or a **TV Show** (`1`). The solution is integrated into an interactive Streamlit web dashboard for real-time model evaluation.

## b. Dataset Description
* **Dataset**: Netflix Movies and TV Shows Dataset
* **Source**: Kaggle / Netflix
* **Instances**: 8797 valid samples (after feature extraction and dropping missing values)
* **Features**: 5 engineered numerical attributes (Release Year, Added Year, Description Length, Cast Count, Listed In Count)
* **Target**: Binary classification (`0` = Movie, `1` = TV Show)

## c. GitHub Repository Link
`https://github.com/2025NS04135/ML-Assignment-2-BITS`

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6983 | 0.6442 | 0.6667 | 0.0075 | 0.0148 | 0.0463 |
| **Decision Tree** | 0.6608 | 0.5996 | 0.4357 | 0.4071 | 0.4210 | 0.1817 |
| **KNN** | 0.7153 | 0.7134 | 0.5412 | 0.3940 | 0.4560 | 0.2759 |
| **Naive Bayes** | 0.7256 | 0.6442 | 0.6157 | 0.2495 | 0.3551 | 0.2547 |
| **Support Vector Machine** | 0.7358 | 0.7243 | 0.7073 | 0.2176 | 0.3329 | 0.2822 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Struggles with this specific non-linear engineered feature set, resulting in very low recall (0.0075) for TV Shows. |
| **Decision Tree** | Offers a more balanced recall (0.4071) compared to linear models by splitting on non-linear thresholds like cast size and categories. |
| **KNN** | Strongest overall accuracy (0.7153) for non-SVM models. Standardizing features allowed it to capture density clusters of Movies vs TV Shows reasonably well. |
| **Naive Bayes** | High accuracy (0.7256) but poor recall, suggesting the independence assumption between features like `release_year` and `added_year` hinders identifying TV Shows. |
| **Support Vector Machine** | Achieved the highest overall accuracy (0.7358) and AUC (0.7243), leveraging the RBF kernel to draw non-linear boundaries around complex feature relationships. |
| **Overall Winner for your dataset?** | **Support Vector Machine**. It yields the best Accuracy and AUC score, efficiently separating movies from TV shows using non-linear margins in the engineered feature space. |
