# ❤️ CardioPredict – Heart Disease Risk Prediction

**CardioPredict** is an advanced machine learning project that integrates a complete ML pipeline into an interactive web application. The goal is to predict the risk of cardiovascular diseases using clinical data, while offering transparency through visual explanations.

---

## 📊 Overview

This project includes:
- A **Jupyter Notebook** implementing a full ML pipeline from preprocessing to SHAP-based model interpretation.
- A **Streamlit-based web app** for interactive heart disease prediction and analysis.

---
## Demo Video
Check out the presentation of the interface and key features here:  
[![Watch the demo](https://img.youtube.com/vi/BI8z01Uph1U?si=yP70yOXioS6ldWDq/0.jpg)](https://www.youtube.com/watch?v=BI8z01Uph1U?si=yP70yOXioS6ldWDq)

## 🧠 ML Pipeline (Jupyter Notebook)

### 🔍 Data Preprocessing
- Handling missing values, outliers
- RobustScaler for numerical features
- One-Hot Encoding for categorical features
- Pipelines built using `scikit-learn`

### 🤖 Model Training
- Classifiers: Logistic Regression, SVM, KNN, Random Forest
- Ensemble: Voting & Stacking
- GridSearchCV for hyperparameter tuning
- Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Cross-validation applied

### 📊 Model Explainability
- Global insights: SHAP Summary, Beeswarm, and Dependence Plots
- Local insights: Force Plot, Waterfall, and Decision Plots

---
## 🌐 Web Application (Streamlit)

### 🔹 Single Prediction
Input patient health parameters and receive a binary prediction (0 = low risk, 1 = high risk), along with **SHAP visualizations**.

### 🔹 Mass Prediction (CSV Upload)
Upload a CSV file containing patient data. The app processes the file, applies transformations, and returns a CSV with prediction results.

### 🔹 Interactive Visualizations
Use dynamic plots (histograms, boxplots, pie charts, radar charts) to compare user inputs with population data.

### 🔹 Model Interpretability
Built-in SHAP Explainer provides feature importance and local/global explanations using:
- Summary Plots
- Dependence Plots
- Waterfall Plots

---

## Technologies & Tools
- **Languages:** Python, HTML, CSS  
- **Libraries:** scikit-learn, SHAP, Pandas, NumPy, Plotly, Matplotlib, Seaborn  
- **Web Framework:** Streamlit  
- **Environment:** Jupyter Notebook / Google Colab  

## 📁 Project Structure
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


## ▶️ Running the Project

### 🔄 Streamlit Web App:
```bash
streamlit run app.py
```


---
