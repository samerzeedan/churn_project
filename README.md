 📊 Customer Churn Prediction — Machine Learning & Deep Learning Project

This project presents a complete end-to-end churn prediction system using both classical Machine Learning models and several Deep Learning architectures. The goal is to identify customers at risk of churn using the Telco Customer Churn dataset.

The workflow includes preprocessing, feature engineering, modeling, hyperparameter tuning, and experiment tracking.

---

## 🚀 Project Highlights

### 1. Full Preprocessing Pipeline

- Handling missing values using:
  - SimpleImputer
  - Advanced Imputers (KNN, Iterative)
- Outlier detection and mitigation
- Categorical encoding
- Robust scaling to handle skewed distributions
- SMOTE for balancing classes
- Additional feature engineering:
  - `TotalServicesCount`
  - `AutomaticPayment`
  - `IsNewCustomer`
  - `TenureServicesScore`
  - `InternetType`
  - Service interactions and derived metrics

---

## 🤖 Machine Learning Models

The following classical ML models were implemented and evaluated:

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble model with decision trees |
| KNN | Distance-based classifier |
| SVM | Margin-based classifier with kernels |
| Decision Tree | Simple rule-based classifier |
| Gradient Boosting | Boosted tree ensemble |
| XGBoost | High-performance gradient boosting |

Model performance was evaluated using:

- Stratified 5-Fold Cross-Validation  
- Accuracy  
- ROC-AUC  
- Log Loss  

---

## 🧠 Deep Learning Models

The project includes several deep learning architectures built and optimized via Optuna:

| Model Type | Description |
|------------|-------------|
| **MLP (Multi-Layer Perceptron)** | Dense neural network with BatchNorm & Dropout. |
| **1D CNN** | Captures local patterns in reshaped tabular sequences. |
| **Simple RNN** | Learns short sequential dependencies. |
| **LSTM** | Learns long-term dependencies with memory gating. |
| **Transformer Encoder** | Uses attention to model global context. |

### Deep Learning Hyperparameters Tuned

| Hyperparameter | Range |
|----------------|--------|
| Learning Rate | 1e-4 → 5e-3 |
| L2 Regularization | 1e-6 → 1e-3 |
| Dropout Rate | 0.0 → 0.4 |
| Dense Units | 32–256 |
| CNN Filters | 16–128 |
| Kernel Sizes | 2–5 |
| Attention Heads | 2–4 |
| d_model | 32–96 |
| FFN Dim | 64–256 |

### 🏆 Best Deep Learning Model

The **MLP Neural Network** achieved the highest overall performance (AUC, accuracy, stability), and was selected as the final deployed model.

---

## 📈 Performance Summary

| Category | Best Model |
|----------|------------|
| Best ML Model | XGBoost |
| Best DL Model | **MLP (Winner)** |
| Final Output | Churn probability |

---

## 📁 Project Structure

churn_project/
│── README.md
│── requirements.txt
│── demo_basic.py
│
├── data/
│ ├── raw_dataset.csv
│ ├── processed_churn.csv
│
├── models/
│ ├── final_xgb_pipeline.pkl
│ ├── best_dl_model.keras
│ ├── scaler.pkl
│ ├── feature_names.pkl
│
├── notebooks/
│ ├── preprocess.ipynb
│ ├── model_ml.ipynb
│ ├── model_dl.ipynb
│ ├── optuna_search.ipynb



---

## 📬 Contact

- LinkedIn: (www.linkedin.com/in/samer-zaidan-60bb372b0)
---
