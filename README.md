# ML Classification Pipeline — ONNX & PMML Deployment

End-to-end machine learning pipeline for 3-class prediction 
on 1.2M records, with dual model deployment in ONNX and PMML formats.

## Overview
A complete ML pipeline built for a large-scale classification problem,
covering data preprocessing, feature engineering, model training,
hyperparameter tuning, and production-ready model deployment.

**Dataset:** 1.2M records · 15 numerical features · 3-class target  
**Model:** Random Forest with RandomizedSearchCV tuning  
**Accuracy:** 73% · AUC-ROC: 0.90 (Class 2)  
**Deployment:** ONNX (edge/real-time inference) + PMML (enterprise platforms)

## Pipeline Steps

### 1. Data Preprocessing
- Missing value imputation (SimpleImputer, mean strategy)
- Z-score outlier detection and removal (threshold = 3σ)
- StandardScaler normalization
- Duplicate row elimination

### 2. Exploratory Data Analysis
- Feature distribution histograms and boxplots
- Correlation heatmap (16×16)
- Class distribution analysis
- Violin plots stratified by class

### 3. Feature Engineering
- PCA dimensionality reduction (98% variance retained)
- SMOTE oversampling to address class imbalance (11.8% minority)
- SelectKBest feature selection

### 4. Model Training
- Random Forest Classifier
- RandomizedSearchCV hyperparameter tuning (5-fold CV, 10 candidates)
- Best params: 108 estimators, max_depth=16, class_weight='balanced'

### 5. Evaluation
- Accuracy, Precision, Recall, F1-Score per class
- Confusion matrix visualization
- ROC curve (AUC per class)

### 6. Model Deployment
- ONNX export via skl2onnx — cross-platform edge inference
- PMML export via sklearn2pmml — enterprise platform integration
- Prediction parity validation (ONNX vs original model)

## Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 1 (minority) | 0.60 | 0.56 | 0.53 |
| 2 (majority) | 0.75 | 1.00 | 0.85 |
| 3 | 0.80 | 0.68 | 0.73 |
| **Overall** | | **0.73** | |

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/ml-classification-pipeline.git
cd ml-classification-pipeline
pip install -r requirements.txt
```

## Usage

Open the notebooks in Jupyter or Google Colab:

- `ONNX.ipynb` — Full pipeline with ONNX deployment
- `PMML.ipynb` — Full pipeline with PMML deployment

> **Note:** The dataset is not included in this repo (1.2M records, 
> proprietary course data). The notebooks can be run with any 
> similarly structured CSV with numerical features and a Class column.

## Tech Stack
Python · scikit-learn · ONNX · PMML · Pandas · NumPy  
SMOTE · PCA · Matplotlib · Seaborn · Google Colab

## Course
IIT Chicago — CSP574 Data Preparation & Analysis (Dec 2024)
