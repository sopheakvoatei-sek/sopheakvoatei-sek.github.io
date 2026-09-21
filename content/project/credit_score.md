
---
title: "Credit Score Analysis & Risk Management (Python)"
date: "2025-12-31"
excerpt: "End-to-end credit risk analysis using financial and behavioral data from 1,000 customers. Applied statistical methods and machine learning to predict customer default and credit scores, achieving an R² of 0.7735 with Random Forest regression."
tags: ["Python", "Machine Learning", "Credit Risk", "Logistic Regression", "Random Forest", "XGBoost", "PCA", "Classification", "Regression"]
thumbnail: "/images/projects/credit_score/credit_score_distribution.png"
---

**ENSIIE – Modeling & Regularized Regression (MRR) Project (2025–2026)**

---

## Project Overview

This project was completed as part of the *Modeling & Regularized Regression (MRR)* course at ENSIIE.

The objective was to investigate two questions:

> Can we predict whether a customer will default based on their financial and behavioral information?

> Can we accurately estimate a customer's credit score using statistical and machine learning models?

Using a dataset of **1,000 customers with 84 financial and behavioral features**, I explored an end-to-end machine learning workflow combining data preprocessing, feature engineering, statistical analysis, and predictive modeling.

The project focuses on two tasks: **default classification** and **credit score regression**, with an emphasis on model comparison, interpretability, and the practical implications of credit risk prediction.

---

## 1. Methodology & Machine Learning Pipeline

The dataset contains financial information, transaction histories, spending patterns, and financial ratios describing customer behavior.

The analysis followed a structured workflow:

**Data Exploration → Feature Engineering → Preprocessing → Statistical Analysis → Predictive Modeling → Evaluation**

The main steps included:

- **Data preparation:** Missing-value analysis, outlier detection, categorical encoding, and feature scaling.
- **Feature engineering:** Financial ratios such as debt-to-income and savings-to-income.
- **Statistical analysis:** PCA, Canonical Correlation Analysis (CCA), and Correspondence Analysis.
- **Classification:** Logistic Regression, Random Forest, and XGBoost for predicting customer default.
- **Regression:** Linear Regression, Random Forest, and XGBoost for estimating credit scores.

The models were evaluated using a 70/30 train-test split, with stratification for the classification task.

---

## 2. Exploratory Data Analysis

Before modeling, I explored the dataset to understand customer financial behavior and identify potential challenges.

### Credit Score Distribution

![Credit Score Distribution](/images/projects/credit_score/credit_score_distribution.png)

The distribution analysis examines the variability of credit scores and their relationship with customer default status.

It provides an initial understanding of the target variable and potential differences between customer risk groups.

### Default Class Balance

![Default Class Balance](/images/projects/credit_score/default_class_balance.png)

The dataset contains more non-defaulting customers than defaulting customers.

This imbalance introduces an important challenge: a model can achieve relatively high accuracy while failing to identify customers who actually default.

Therefore, **recall, F1-score, and ROC-AUC** were considered alongside accuracy when evaluating classification models.

### Feature Correlation Analysis

![Feature Correlation Analysis](/images/projects/credit_score/feature_correlation.png)

Correlation analysis was used to investigate relationships between financial indicators and the target variables.

Financial ratios and spending-related features emerged as relevant indicators, highlighting the importance of considering relationships between income, debt, savings, and expenditure rather than relying only on absolute monetary values.

---

## 3. Principal Component Analysis (PCA)

Financial variables are often strongly correlated, introducing redundancy into the feature space.

I investigated Principal Component Analysis to determine whether dimensionality reduction could preserve the main structure of the data while improving predictive performance.

### Explained Variance

![PCA Explained Variance](/images/projects/credit_score/cumulative_explain_var.png)

Using a cumulative explained variance threshold of 90%, PCA retained:

**18 principal components, explaining approximately 90.4% of total variance.**

Although PCA substantially reduced the number of features, its effectiveness for prediction needed to be evaluated separately.

Models trained on the original features were therefore compared with models trained on the PCA-transformed data.

---

## 4. Default Prediction — Classification

The first predictive task was to identify customers who may default.

Three classification models were evaluated:

- **Logistic Regression:** An interpretable linear model for estimating default probability.
- **Random Forest:** An ensemble model capable of capturing non-linear relationships.
- **XGBoost:** A gradient-boosting model combining decision trees.

Class weighting was applied to Logistic Regression and Random Forest to address the imbalance between defaulting and non-defaulting customers.

### Classification Results

| Model | Accuracy | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 66.67% | 51.76% | 0.4681 | 0.6256 |
| Random Forest | 70.67% | 9.41% | 0.1538 | 0.6174 |
| XGBoost | 71.33% | 21.18% | 0.2951 | 0.6137 |

### Model Comparison & ROC Curves

![Classification Model Comparison](/images/projects/credit_score/classification_model_compare.png)

The results reveal an important trade-off between overall accuracy and the ability to detect defaulters.

**Logistic Regression achieved the highest recall (51.76%) and F1-score (0.4681)** among the tested models.

Although Random Forest and XGBoost achieved higher overall accuracy, they missed a larger proportion of actual defaulters.

This demonstrates why accuracy alone is insufficient for evaluating credit risk classification.

However, the ROC-AUC values of approximately 0.61–0.63 also indicate limited discriminative performance, suggesting that further improvements would be necessary before practical deployment.

---

## 5. Credit Score Prediction — Regression

The second task was to predict customers' numerical credit scores.

Three regression models were compared:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Performance was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

### Regression Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 21.64 | 29.68 | 0.7469 |
| Random Forest | 20.59 | 28.08 | 0.7735 |
| XGBoost | 21.68 | 29.53 | 0.7495 |

### Model Comparison

![Regression Model Comparison](/images/projects/credit_score/regression_model_compare.png)

**Random Forest achieved the strongest regression performance**, with:

- R² = 0.7735
- MAE = 20.59
- RMSE = 28.08

The model explained approximately 77.35% of the variation in credit scores on the test set, with an average absolute prediction error of approximately 20.6 points.

These results suggest that non-linear relationships and interactions between financial variables provide useful predictive information.

However, Linear Regression also achieved a relatively strong R² of 0.7469, indicating that linear relationships explain a substantial portion of the variation in credit scores.

### Feature Importance

![Random Forest Feature Importance](/images/projects/credit_score/features_predict_credit_score.png)

Random Forest feature importance was examined to identify which financial indicators contributed most to credit score prediction.

The results highlight the relevance of financial ratios and expenditure-related variables.

This reinforces the importance of feature engineering and shows how relationships between financial variables can provide useful predictive information.

---

## 6. Does PCA Improve Predictive Performance?

To evaluate the effectiveness of dimensionality reduction, I compared models trained on the original features with models trained using 18 principal components.

### Classification with PCA

![PCA Classification Comparison](/images/projects/credit_score/PCA_classification_model_compare.png)

PCA slightly improved Logistic Regression's ROC-AUC from 0.6256 to 0.6530 and recall from 51.76% to 54.12%.

However, its accuracy and F1-score decreased, while XGBoost also experienced a reduction in ROC-AUC.

### Regression with PCA

![PCA Regression Comparison](/images/projects/credit_score/PCA_regression_model_compare.png)

PCA substantially reduced regression performance.

For example, Linear Regression's R² decreased from 0.7469 to 0.6058, while XGBoost's R² decreased from 0.7495 to 0.6113.

**Key insight: Preserving variance does not necessarily preserve predictive information.**

Although PCA retained approximately 90% of the dataset's variance, some information relevant to predicting the target variables was not adequately preserved.

This demonstrates why dimensionality reduction should be evaluated using downstream predictive performance rather than explained variance alone.

---

## 7. Key Findings

The project highlighted four important lessons.

**1. Accuracy is not sufficient for credit risk classification.**

Logistic Regression achieved the highest recall for identifying defaulters, while Random Forest and XGBoost achieved higher accuracy but missed more defaulting customers.

**2. Non-linear models can improve credit score prediction.**

Random Forest achieved an R² of 0.7735 and an MAE of 20.59, outperforming the other regression models in this experiment.

**3. Financial ratios provide valuable predictive information.**

Relationships between debt, savings, income, and expenditure help capture customer financial behavior beyond individual monetary values.

**4. Dimensionality reduction does not guarantee better performance.**

PCA retained approximately 90% of total variance using 18 components but substantially reduced regression performance.

These findings demonstrate the importance of balancing statistical analysis, feature engineering, model complexity, and evaluation metrics.

---

## Technical Stack

| Category | Tools & Methods |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Statistical Analysis | PCA, CCA, Correspondence Analysis |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Accuracy, Recall, F1, ROC-AUC, MAE, RMSE, R² |

---

## Conclusion

This project demonstrates an end-to-end exploratory machine learning workflow for credit risk analysis, combining statistical methods, financial feature engineering, classification, and regression.

The main takeaway is that **model selection depends on the prediction objective**: identifying potential defaulters requires different considerations from estimating numerical credit scores.

The experience strengthened my understanding of how to translate financial data into meaningful predictive insights while balancing model performance, interpretability, and practical risk considerations.

---

## Explore the Full Analysis

Interested in the technical details behind this project?

Explore the complete methodology, statistical analysis, model implementation, and additional visualizations.

[**View Full Technical Report →**](/project/credit_score_detailed/)