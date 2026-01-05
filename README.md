<div align="center">

# ❤️ CardioPredict
### Heart Disease Risk Assessment System using Explainable AI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn)
![Bachelor's Thesis](https://img.shields.io/badge/Project-Bachelor's_Thesis-blue?style=for-the-badge)

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#installation">Installation</a> •
  <a href="#model-performance">Model Performance</a>
</p>

</div>

---

## 🎬 Project Demo
Watch the full presentation of the interface and key features on YouTube:

[![Watch the demo](https://img.youtube.com/vi/BI8z01Uph1U/maxresdefault.jpg)](https://www.youtube.com/watch?v=BI8z01Uph1U)

---

## 📖 Overview

**CardioPredict** is an end-to-end Machine Learning solution designed to predict the risk of cardiovascular diseases based on clinical data. Unlike standard "black-box" models, this project prioritizes **transparency and interpretability** by integrating **SHAP (SHapley Additive exPlanations)** values.

The system consists of two main components:
1.  **Backend ML Pipeline:** rigorous data preprocessing, model selection, hyperparameter tuning, and evaluation.
2.  **Frontend Web App:** An interactive Streamlit dashboard allowing doctors and users to input data (single or batch) and receive understandable risk assessments.

---

## ML Pipeline (Jupyter Notebook)

### Data Preprocessing
- Handling missing values, outliers
- RobustScaler for numerical features
- One-Hot Encoding for categorical features
- Pipelines built using `scikit-learn`

### Model Training
- Classifiers: Logistic Regression, SVM, KNN, Random Forest
- Ensemble: Voting & Stacking
- GridSearchCV for hyperparameter tuning
- Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Cross-validation applied

### Model Explainability
- Global insights: SHAP Summary, Beeswarm, and Dependence Plots
- Local insights: Force Plot, Waterfall, and Decision Plots

---
## Web Application (Streamlit)

### Single Prediction
Input patient health parameters and receive a binary prediction (0 = low risk, 1 = high risk), along with **SHAP visualizations**.

### Batch Prediction
(CSV & Excel) Upload a CSV or Excel file containing patient data. The app processes the file, applies transformations, and returns the prediction results (available for download in both CSV and Excel formats). 
### Interactive Visualizations
Use dynamic plots (histograms, boxplots, pie charts, radar charts) to compare user inputs with population data.

### Model Interpretability
Built-in SHAP Explainer provides feature importance and local/global explanations using:
- Summary Plots
- Dependence Plots
- Waterfall Plots

---

## Technologies & Tools
| Category | Technologies |
| :--- | :--- |
| **Core** | Python 3.x |
| **Machine Learning** | scikit-learn, NumPy, Pandas |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Framework** | Streamlit |
| **Environment** | Jupyter Notebook, Anaconda |


## Project Structure
```
├── app.py                  # Streamlit App
├── requirements.txt        # Python dependencies
├── Prediction/
│   ├── model.pkl           # Trained model
│   ├── pipeline.pkl        # Preprocessing pipeline
│   └── explainer.pkl       # SHAP explainer
├── data/
│   └── heart.csv        # Oryginal dataset
├── assets/
│   └── style.css           # Custom styles
├── CardioPredict.ipynb     # Jupyter notebook with ML pipeline
```

---


## Running the Project

### Streamlit Web App:
```bash
streamlit run app.py
```

