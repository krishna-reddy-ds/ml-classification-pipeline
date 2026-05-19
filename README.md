# ML Classification Pipeline — ONNX & PMML Deployment

End-to-end machine learning pipeline for 3-class prediction on 1.2M records,  
with model deployment in ONNX and PMML formats.

## Overview
- **Dataset:** 1.2M records, 15 numerical features, 3-class target
- **Model:** Random Forest with RandomizedSearchCV tuning
- **Accuracy:** 73% | AUC-ROC: 0.90
- **Deployment:** ONNX (edge inference) + PMML (enterprise platforms)

## Pipeline Steps
1. Data loading & EDA
2. Outlier removal (Z-score, threshold=3σ)
3. StandardScaler normalization
4. SMOTE oversampling (11.8% minority class)
5. PCA dimensionality reduction (98% variance retained)
6. Random Forest + RandomizedSearchCV (5-fold CV)
7. Model serialization — ONNX & PMML
8. Prediction parity validation

## Tech Stack
Python · scikit-learn · ONNX · PMML · Pandas · NumPy · SMOTE · Matplotlib · Seaborn

## Course
IIT Chicago — CSP574 Data Preparation & Analysis (Dec 2024)
