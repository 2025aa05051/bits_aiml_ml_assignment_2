# ML Assignment 2

## Problem Statement
The goal of this project is to build and compare multiple supervised machine learning classification models to predict whether a bank client will subscribe to a term deposit (`y`). The project includes model training, evaluation using multiple metrics, and deployment of an interactive Streamlit app to demonstrate model performance on uploaded test data.

---

## Dataset Description
- **Dataset Name:** Bank Marketing (bank-additional-full)
- **Type:** Binary Classification (`y`: yes/no)
- **Records:** 45,211 instances (public dataset)
- **Features:** 16 input features (mix of numerical and categorical)
- **Target Column:** `y` (mapped to 1 for "yes" and 0 for "no")
- **Source:** UCI Machine Learning Repository
- **Source Link :** https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## Models Implemented
The following models were trained on the same training set and evaluated on the same test set:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

## Model Comparison Table (Evaluation Metrics)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| KNN | 0.8986 | 0.8500 | 0.6257 | 0.3318 | 0.4336 | 0.4070 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | 0.9073 | 0.9291 | 0.6698 | 0.4102 | 0.5088 | 0.4778 |
| XGBoost | **0.9103** | **0.9334** | 0.6598 | 0.4820 | **0.5571** | **0.5163** |

---

## Observations (Model-wise)
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with good AUC (0.9056) and precision (0.6445), but lower recall (0.3478). This indicates it is conservative in predicting the positive class (subscription). |
| Decision Tree | Balanced precision/recall compared to LR and KNN, but noticeably lower AUC (0.7015). This suggests overfitting or weaker ranking ability on probabilities. |
| KNN | Performs close to Logistic Regression in accuracy, but recall is low (0.3318). Likely affected by feature scaling and class imbalance, where nearest neighbors are dominated by majority class. |
| Naive Bayes | Higher recall (0.5198) but low precision (0.4059). It catches more positive cases but introduces more false positives, which is typical due to its independence assumption. |
| Random Forest | Strong overall performance with higher AUC (0.9291) and improved recall vs LR/KNN. Ensemble averaging reduces overfitting and improves generalization. |
| XGBoost | Best overall model: highest accuracy (0.9103), AUC (0.9334), F1 (0.5571), and MCC (0.5163). It provides the best balance between precision and recall, making it the most reliable for this dataset. |

---

## Repository Structure
- `model/` contains separate python files for each model and training orchestration.
- `model/artifacts/` contains saved trained pipelines for Streamlit inference.
- `data/test.csv` is generated from the dataset split and is used for Streamlit upload.
- `datapreperation` contains configuration, preprocessor & metric informations

---

## Streamlit Application
### Features implemented (as per assignment):
- Upload test CSV file
- Select model from dropdown
- Display evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Display confusion matrix and classification report

---

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt


