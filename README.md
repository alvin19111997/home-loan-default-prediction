# home-loan-default-prediction
A machine learning project to predict home loan default risk using applicant financial data, bureau reports, and previous credit history. Includes data preprocessing, feature engineering, and model evaluation.
---

## 🔍 Problem Statement

Banks want to assess the risk of loan applicants defaulting on their home loans. This project uses historical loan application data along with supplementary sources like previous applications and bureau data to build a predictive model.

---

## 🧰 Tools & Technologies

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook
- XGBoost / Logistic Regression
- Git & GitHub

---

## 🧼 Key Data Cleaning Steps

- Handled missing values using median/mode imputation
- Dropped highly null or redundant features
- Winsorized extreme outliers (capped at 5%)
- Removed multicollinear features (correlation > 0.85)
- Encoded categorical variables (ordinal, one-hot, frequency)

---

## ⚖️ Class Imbalance Handling

Only ~8% of applicants were defaulters. To prevent bias:
- Stratified train-test split
- Applied `class_weight='balanced'` during modeling

---

## 📈 Modeling

Tested the following algorithms:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (best performance)

Metrics evaluated:
- Accuracy
- F1 Score
- ROC-AUC

---

## ✅ Final Result

Achieved **balanced performance** with ROC-AUC > 0.78 using XGBoost on the cleaned and engineered feature set.

---
