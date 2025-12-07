# churn_project
End-to-end Machine Learning &amp; Deep Learning project for predicting customer churn using Telco dataset.
# 📊 Customer Churn Prediction — Machine Learning & Deep Learning Project

This project presents a complete **end-to-end churn prediction system** built using both classical Machine Learning models and a Deep Learning (MLP) model. The solution is applied to the Telco Customer Churn dataset and aims to accurately identify customers who are at risk of leaving.

The project includes data preprocessing, feature engineering, model training, hyperparameter optimization, experiment tracking, and deployment-ready prediction scripts.

---

## 🚀 Project Highlights

### ✔️ 1. Complete Preprocessing Pipeline
A full, production-ready preprocessing workflow including:

- **Handling Missing Values**
  - SimpleImputer (mean, median, most_frequent)
  - Advanced Imputation (KNN Imputer, Iterative Imputer)
- **Encoding categorical variables**
- **Scaling with RobustScaler** to handle outliers
- **Outlier detection and treatment**
- **Feature Engineering**
  - `TotalServicesCount`
  - `AutomaticPayment`
  - `IsNewCustomer`
  - `TenureServicesScore`
  - `InternetType` mapping  
  - Interaction features between tenure and services
- Dataset balancing using **SMOTE**

---

### ✔️ 2. Machine Learning Models Implemented

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | High-performing ensemble method |
| KNN | Distance-based algorithm |
| SVM | Non-linear classifier with kernels |
| Decision Tree | Simple interpretable model |
| Gradient Boosting | Sequential boosting algorithm |
| XGBoost | Optimized boosting algorithm |

All models were evaluated using:
- **Stratified 5-Fold Cross Validation**
- **AUC**, **Accuracy**, **Log Loss**, **Confusion Matrix**

---

### ✔️ 3. Deep Learning Model (MLP Neural Network)
A fully-connected neural network built with TensorFlow/Keras featuring:

- Multiple Dense layers  
- ReLU activation functions  
- Dropout layers  
- Batch Normalization  
- Adam optimizer  
- Early stopping  

This model achieved the **highest AUC and accuracy** across all experiments—making it the final selected model.

---

### ✔️ 4. Hyperparameter Tuning with Optuna

Optuna was used to automatically search for the **optimal hyperparameters** for all ML models:

- Logistic Regression (C)
- Random Forest (depth, estimators, min samples)
- KNN (n_neighbors, weights)
- SVM (C, gamma)
- Decision Tree (max_depth, criterion)
- Gradient Boosting (estimators, learning rate)
- XGBoost (depth, subsample, colsample_bytree, learning_rate)

The optimization objective was **maximize Validation AUC**.

---

### ✔️ 5. Experiment Tracking with MLflow
MLflow tracked:

- Model parameters  
- Training metrics  
- Cross-validation scores  
- Loss curves  
- AUC values  
- Saved model artifacts  

This allows full reproducibility and experiment comparison.

---

### ✔️ 6. Deployment-Ready Prediction Script
The project includes a ready-to-use script:

`demo_basic.py`

It performs:
- Loading scaler  
- Loading feature names  
- Loading ML or DL final model  
- Converting input into correct format  
- Predicting churn probability  

---

## 📁 Project Structure

churn_project/
│── README.md
│── requirements.txt
│── demo_basic.py
│── mlflow.db (optional)
│
├── data/
│ ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│ ├── processed_churn.csv
│
├── models/
│ ├── final_xgb_pipeline.pkl
│ ├── feature_names.pkl
│ ├── scaler.pkl
│ ├── best_dl_model.keras
│
├── notebooks/
│ ├── preprocess.ipynb
│ ├── model_ml.ipynb
│ ├── model_dl.ipynb
│ ├── best_deep_learning.ipynb

yaml


---

## 📈 Model Performance

| Model | AUC | Accuracy |
|-------|------|-----------|
| Logistic Regression | ... | ... |
| Random Forest | ... | ... |
| Gradient Boosting | ... | ... |
| XGBoost | ... | ... |
| **Deep Learning (MLP)** | **Highest** | **Highest** |

*(Add your final numbers here)*

---

## 🧰 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Optuna  
- MLflow  
- Imbalanced-Learn (SMOTE)  
- Matplotlib / Seaborn  

---

## ▶️ How to Run

### Install dependencies:
pip install -r requirements.txt

shell

### Run prediction demo:
python demo_basic.py

yaml


---

## 📬 Contact

If you'd like to connect or discuss the project:

- **LinkedIn:** (
www.linkedin.com/in/samer-zaidan-60bb372b0

)

---
