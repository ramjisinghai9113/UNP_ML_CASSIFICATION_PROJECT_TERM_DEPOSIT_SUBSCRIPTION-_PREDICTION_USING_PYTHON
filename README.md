# UNP_ML_CASSIFICATION_PROJECT_TERM_DEPOSIT_SUBSCRIPTION-_PREDICTION_USING_PYTHON
This project was completed as part of the United Network of Professionals (UNP) Machine Learning Academic Internship Program.

# 🏦 Bank Marketing Classification: Predicting Term Deposit Subscription

An end-to-end machine learning project that predicts whether a bank customer will subscribe to a **term deposit** after a direct marketing (phone) campaign. It covers data cleaning, feature engineering, exploratory data analysis, class-imbalance handling with SMOTE, a comparison of five classical ML models, and an Artificial Neural Network (ANN) with hyperparameter search.

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Key EDA Insights](#-key-eda-insights)
- [Models and Results](#-models-and-results)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Repository Structure](#-repository-structure)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Problem Statement
Banks run large telemarketing campaigns, but only a small share of contacted customers subscribe to a term deposit. The goal is to build a classifier that identifies customers likely to subscribe, so that marketing effort can be focused on the right people.

- **Task:** Binary classification
- **Target (`y`):** `yes` (subscribed) / `no` (did not subscribe)

## 📂 Dataset
- **Source:** Bank Marketing dataset (`bank-full`), a Portuguese bank's direct marketing campaigns (UCI Machine Learning Repository)
- **Size:** 45,211 rows × 17 columns (43,908 rows after cleaning)
- **Class balance (after cleaning):** 88.4% *No* vs 11.6% *Yes*, a highly imbalanced target

| Type | Features |
|------|----------|
| Numerical | `age`, `balance`, `campaign`, `pdays`, `previous` |
| Categorical | `marital`, `education`, `default`, `housing`, `loan`, `contact`, `poutcome`, `income_group` (derived from `job`) |

## 🔄 Project Workflow

### 1. Data Preprocessing
- Dropped `month`, `day` and `duration`. `duration` is only known after the call ends, so using it would leak the outcome into the model.
- **Feature engineering:** grouped `job` into an `income_group` feature (High / Mid / Low), then dropped `job`.
  - High: management, entrepreneur, self-employed
  - Mid: technician, admin., services, retired
  - Low: blue-collar, housemaid, student, unknown
- Removed 1,303 rows with no income group (jobs not in the mapping), leaving **43,908** records with no missing values.
- Binary encoding for `default`, `housing`, `loan` and `y`; one-hot encoding (`drop_first=True`) for `income_group`, `marital`, `education`, `contact` and `poutcome`.

### 2. Exploratory Data Analysis
Histograms with KDE, IQR-based outlier counts, boxplots, count plots, pie charts, subscription rates per category, and a correlation heatmap.

### 3. Handling Class Imbalance
- Stratified train/test split with `random_state=42`
- **SMOTE** applied on the training data only, which balances the classes to 50/50 (31,056 each at an 80/20 split)
- `StandardScaler` fitted on the training data and applied to the test data

### 4. Modelling
- Five ML models evaluated across four train/test splits (80/20, 75/25, 70/30, 60/40)
- An ANN tuned over architecture, batch size and optimizer

## 🔍 Key EDA Insights
- **Previous campaign outcome is the strongest signal:** customers with `poutcome = success` subscribed **64.1%** of the time, against 9.1% when the outcome was unknown.
- **Contact channel:** customers with an `unknown` contact type subscribed at only **4.0%**, versus 14.8% for cellular.
- **Housing loan:** customers *without* a housing loan subscribed more (16.5% vs 7.7%). The same is true for those without a personal loan (12.5% vs 6.7%).
- **Education:** subscription rises with education level (primary 8.5%, secondary 10.4%, tertiary 14.9%).
- **Marital status:** single customers subscribe most (14.8%), married least (10.0%).
- **Income group:** the Low group subscribes least (9.2%) versus High (12.9%) and Mid (12.3%).
- **Credit default:** customers in default subscribe far less (6.4% vs 11.7%).
- **Outliers:** `balance`, `campaign`, `pdays` and `previous` are heavily right-skewed with many IQR outliers.

## 🤖 Models and Results

### Classical ML (SMOTE + scaling)
Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost.

| Model | Test Split | Precision | Recall | F1 | Accuracy |
|-------|-----------|-----------|--------|----|----------|
| Random Forest (highest accuracy) | 40% | 0.322 | 0.334 | 0.328 | **84.13%** |
| Gradient Boosting (highest F1) | 20% | 0.302 | 0.417 | **0.350** | 82.10% |
| XGBoost | 20% | 0.313 | 0.394 | 0.349 | 82.94% |
| Logistic Regression | 20% | 0.257 | 0.382 | 0.307 | 80.04% |
| Decision Tree | 20% | 0.223 | 0.333 | 0.267 | 78.82% |

The Random Forest was retrained on the 80/20 split and evaluated with a confusion matrix and classification report (accuracy 84.05%).

### Artificial Neural Network (Keras)
- Architecture: Dense (ReLU) → Dropout (0.2) → Dense (ReLU) → Dropout (0.2) → Dense (sigmoid)
- Grid search: hidden units `(8,4)`, `(16,8)`, `(32,16)`; batch size `32/64`; optimizer `Adam/SGD`
- Training: binary cross-entropy, learning rate 0.001, `EarlyStopping` on validation loss (patience 5)

| Best configuration | Accuracy | Precision | Recall | F1 |
|--------------------|----------|-----------|--------|----|
| 32-16, Adam, batch 64 (best F1) | 89.43% | 66.54% | 17.60% | 27.84% |
| 32-16, Adam, batch 32 (best accuracy) | **89.46%** | 68.42% | 16.62% | 26.74% |

### 📝 Takeaways
- Because only about 12% of customers subscribe, **accuracy alone is misleading**. A model that always predicts "No" would already score about 88%.
- The ANN reaches the highest accuracy, but with low recall. It was trained on the original imbalanced data, so it favours the majority class.
- The SMOTE-based tree ensembles (Gradient Boosting, XGBoost) give a better precision/recall balance and the best F1 on the positive class.
- The right model depends on the business goal: choose higher recall to reach more potential subscribers, or higher precision to cut wasted calls.

## 🛠 Tech Stack
**Language:** Python
**Data:** Pandas, NumPy
**Visualization:** Matplotlib, Seaborn
**ML:** Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
**Deep Learning:** TensorFlow / Keras
**Environment:** Jupyter Notebook / Google Colab

## ▶️ How to Run
1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn tensorflow openpyxl
   ```
3. Place the dataset (`bank-full.xlsx`) in the project folder. The notebook reads it from `/content/bank-full.xlsx` (the Google Colab path), so update the path in the `pd.read_excel(...)` cell if you run it locally.
4. Open and run `CompletedClassification_Banking_data_UNP_Ramji.ipynb` from top to bottom.

## 📁 Repository Structure
```
├── CompletedClassification_Banking_data_UNP_Ramji.ipynb   # Full analysis and modelling
├── bank-full.xlsx                                         # Dataset (add locally)
└── README.md
```

## 🚀 Future Improvements
- Tune the decision threshold and use precision-recall curves rather than the default 0.5 cut-off
- Apply SMOTE (or class weights) to the ANN to improve recall
- Hyperparameter tuning with cross-validation (GridSearchCV / Optuna)
- Cap or transform skewed features such as `balance` and `campaign`
- Add feature-importance and SHAP analysis for interpretability

Code Link:https://drive.google.com/file/d/1HP6pIDrRErPADFsXc74J8Cf2A7PUeMwi/view?usp=sharing
PPT Link: https://docs.google.com/presentation/d/10-TQH7bFDRWN6Fw5o3F2wfIiecwUEazd/edit?usp=sharing&ouid=115260607757946087302&rtpof=true&sd=true
