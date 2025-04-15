# ❤️ CardioPredict – Heart Disease Risk Prediction

**CardioPredict** is an advanced machine learning project that integrates a complete ML pipeline into an interactive web application. The goal is to predict the risk of cardiovascular diseases using clinical data, while offering transparency through visual explanations.

---

## 📊 Overview

This project includes:
- A **Streamlit-based web app** for interactive heart disease prediction and analysis.
- A **Jupyter Notebook** implementing a full ML pipeline from preprocessing to SHAP-based model interpretation.

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
- Force and Waterfall Plots
- Decision Plots

---

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

## 📁 Project Structure
```
├── app.py                  # Streamlit App
├── requirements.txt        # Python dependencies
├── ML/
│   ├── model.pkl           # Trained model
│   ├── pipeline.pkl        # Preprocessing pipeline
│   └── explainer.pkl       # SHAP explainer
├── data/
│   └── heart_pl.csv        # Oryginal dataset
├── assets/
│   └── style.css           # Custom styles
├── CardioPredict.ipynb     # Jupyter notebook with ML pipeline
```

---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/cardiopredict.git
cd cardiopredict
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 🔄 Streamlit Web App:
```bash
streamlit run app.py
```

### 📃 Jupyter Notebook:
Open `CardioPredict.ipynb` in Jupyter or Colab to run the ML pipeline.

---

## 🚀 Future Enhancements
- Support for additional diseases
- Deep learning modules for imaging
- Expand dashboard using Plotly Dash or Streamlit components

---

## 📨 Contact
**Email:** grzegorz.drozdz@edu.uekat.pl

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements
- SHAP by Scott Lundberg
- Streamlit Community
- Medical datasets via Kaggle

> 🎓 This project is part of my bachelor's portfolio. It demonstrates a full end-to-end ML workflow, including explainable AI, and highlights my interest in Data Science and real-world ML applications.
