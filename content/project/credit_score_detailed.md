---
title: "Credit Score Analysis & Risk Management (Python)"
date: "2025-12-31"
excerpt: "End-to-end credit risk analysis using financial and behavioral data from 1,000 customers. Compared Logistic Regression, Random Forest, and XGBoost for default and credit score prediction, achieving an R² of 0.7735 for credit score regression."
tags: ["Python", "Machine Learning", "Credit Risk", "Logistic Regression", "Random Forest", "XGBoost", "PCA", "Classification", "Regression"]
thumbnail: "/portfolio/assets/images/credit-score/thumbnail.png"
---

**ENSIIE – Modeling & Regularized Regression (MRR) Project (2025–2026)**

---

## Project Overview

This project was completed as part of the *Modeling & Regularized Regression (MRR)* course at ENSIIE.

Credit risk assessment plays an important role in financial institutions, helping lenders evaluate customers' creditworthiness and potential default risk.

The project investigates two fundamental questions:

> Can we predict whether a customer will default based on their financial and behavioral information?

> Can we accurately estimate a customer's credit score using statistical and machine learning models?

Using a dataset of 1,000 customers, I explored an end-to-end machine learning workflow combining statistical analysis, feature engineering, dimensionality reduction, classification, and regression.

The objective was not only to obtain accurate predictions but also to understand which financial indicators contribute to credit risk and how different modeling approaches affect predictive performance.

---

## 1. Dataset Description

The dataset contains information on **1,000 customers with 84 original financial and behavioral features**.

These variables describe different aspects of each customer's financial situation.

| Feature Group | Examples |
|---|---|
| Financial Status | Income, savings, debt |
| Financial Ratios | Debt-to-income, savings-to-income |
| Spending Behavior | Expenditure across 11 categories |
| Transaction History | Spending over 6- and 12-month periods |
| Financial Attributes | Credit cards, mortgages, savings accounts |

Two target variables were investigated.

**DEFAULT — Classification**

A binary variable indicating whether a customer defaults.

- 0: Non-default
- 1: Default

**CREDIT_SCORE — Regression**

A numerical variable representing the customer's credit score.

These two prediction tasks require different modeling strategies and evaluation metrics.

---

## 2. Machine Learning Pipeline

The project follows a structured data science workflow, from raw financial data to predictive modeling and interpretation.

**Step 1 — Data Exploration**

Dataset inspection, descriptive statistics, missing-value analysis, and outlier detection.

↓

**Step 2 — Feature Engineering & Preprocessing**

Financial ratios, categorical encoding, transformations, and scaling.

↓

**Step 3 — Exploratory Data Analysis**

Credit score distribution, default class balance, and feature correlations.

↓

**Step 4 — Statistical Analysis & Dimensionality Reduction**

Principal Component Analysis (PCA), Canonical Correlation Analysis (CCA), and Correspondence Analysis.

↓

**Step 5 — Predictive Modeling**

Classification models for default prediction and regression models for credit score estimation.

↓

**Step 6 — Evaluation & Interpretation**

Model comparison, ROC curves, feature importance, and prediction error analysis.

---

## 3. Exploratory Data Analysis

Before building predictive models, I explored the dataset to understand customer financial behavior, identify potential outliers, and examine the distribution of the target variables.

### 3.1 Credit Score Distribution

The first step was to examine the distribution of credit scores and how they differ between defaulting and non-defaulting customers.

![Credit Score Distribution](/images/projects/credit_score/credit_score_distribution.png)

The visualization provides an initial understanding of the target variable, including its distribution, variability, and relationship with customer default status.

It also helps identify potential outliers and differences between the two customer groups.

### 3.2 Default Class Balance

An important challenge in credit risk modeling is the imbalance between customers who default and those who do not.

![Default Class Balance](/images/projects/credit_score/default_class_balance.png)

The dataset contains a larger proportion of non-defaulting customers.

This imbalance matters because a model can achieve relatively high accuracy by predicting the majority class while failing to identify customers who actually default.

For this reason, classification performance was evaluated using several complementary metrics.

| Metric | Interpretation |
|---|---|
| Accuracy | Proportion of correctly classified customers |
| Precision | Proportion of predicted defaulters who actually default |
| Recall | Proportion of actual defaulters correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to distinguish between the two classes across thresholds |

In credit risk applications, failing to identify a potential defaulter can have financial consequences, making recall an important consideration alongside false-positive rates.

---

## 4. Feature Engineering & Preprocessing

Financial datasets often contain variables with different scales, skewed distributions, and strong correlations.

Several preprocessing techniques were explored to make the data more suitable for statistical analysis and machine learning.

### 4.1 Financial Ratio Features

Financial ratios were used to describe relationships between income, debt, savings, and expenditure.

| Financial Ratio | Interpretation |
|---|---|
| Debt-to-Income | Debt relative to income |
| Debt-to-Savings | Debt relative to available savings |
| Savings-to-Income | Savings relative to income |
| Expenditure-to-Income | Spending relative to income |

These ratios capture relationships between financial variables that absolute monetary values alone may not reveal.

For example, two customers may have the same debt but very different financial situations depending on their income and savings.

### 4.2 Data Transformation

The preprocessing workflow included:

- Checking missing values and duplicate observations.
- Encoding categorical financial attributes.
- Applying logarithmic transformations to skewed numerical variables.
- Using scaling techniques to reduce the influence of extreme values.
- Examining correlations between predictors.

These transformations help address differences in feature distributions and scales.

### 4.3 Feature Correlation Analysis

I examined Pearson correlations to identify relationships between financial indicators and the two target variables.

![Feature Correlation Analysis](/images/projects/credit_score/feature_correlation.png)

The analysis highlights financial ratios and spending-related variables as relevant indicators.

However, correlation alone does not establish causation or guarantee predictive importance.

To investigate which variables contribute to predictions, I also examined feature importance from the trained models.

---

## 5. Principal Component Analysis (PCA)

Financial variables are often strongly correlated, which can introduce redundancy into the feature space.

I investigated Principal Component Analysis (PCA) to determine whether a smaller number of components could preserve the main structure of the financial data.

PCA transforms the original predictors into orthogonal components that capture decreasing amounts of variance.

### 5.1 Explained Variance Analysis

![Explained Variance Analysis](/images/projects/credit_score/cumulative_explain_var.png)

The number of principal components was selected using a cumulative explained variance threshold of 90%.

The analysis retained:

**18 principal components, explaining approximately 90.4% of total variance.**

This substantially reduced the dimensionality of the feature space.

However, an important question remained:

> Does retaining most of the variance also preserve the information needed for accurate predictions?

To investigate this, I trained additional classification and regression models using the PCA-transformed features and compared their performance with models trained on the original feature space.

---

## 6. Default Prediction — Classification

The first predictive task was to classify customers as defaulting or non-defaulting.

I compared three machine learning approaches.

**Logistic Regression**

An interpretable linear classification model that estimates the probability of customer default.

**Random Forest**

An ensemble of decision trees capable of capturing non-linear relationships and interactions between financial variables.

**XGBoost**

A gradient-boosting approach that sequentially combines decision trees to improve predictive performance.

### 6.1 Experimental Setup

The dataset was divided into:

- 70% training data
- 30% testing data

A stratified split was used to preserve the default class proportions.

Class weighting was applied to Logistic Regression and Random Forest to address the imbalance between defaulting and non-defaulting customers.

The models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

### 6.2 Classification Results

The following results were obtained on the 300-customer test set.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 66.67% | 42.72% | 51.76% | 0.4681 | 0.6256 |
| Random Forest | 70.67% | 42.11% | 9.41% | 0.1538 | 0.6174 |
| XGBoost | 71.33% | 48.65% | 21.18% | 0.2951 | 0.6137 |

### 6.3 Model Comparison & ROC Curves

![Model Comparison & ROC curves](/images/projects/credit_score/classification_model_compare.png)

The results reveal an important trade-off between overall accuracy and the ability to detect defaulters.

**Logistic Regression**

Achieved the highest recall and F1-score among the three models.

Its recall of 51.76% indicates that it correctly identified approximately half of the customers who actually defaulted.

**Random Forest**

Achieved higher overall accuracy but detected only 9.41% of actual defaulters.

This illustrates why accuracy alone can be misleading when the target classes are imbalanced.

**XGBoost**

Achieved the highest overall accuracy and precision, but its recall remained relatively low at 21.18%.

### 6.4 Understanding Feature Contributions

To understand the Logistic Regression model, I examined the magnitude and direction of its coefficients.

![Logistic Regression Feature](/images/projects/credit_score/logistic_reg_feature.png)

Positive and negative coefficients indicate the direction of association with the predicted default probability, holding other model inputs fixed.

Because features were scaled, coefficient magnitudes provide a useful perspective on their relative influence within the fitted model.

However, correlated predictors can make individual coefficients difficult to interpret independently.

### Classification Insight

**Logistic Regression provided the highest recall and F1-score among the tested models for identifying potential defaulters.**

The tree-based models achieved higher overall accuracy but missed more customers who actually defaulted.

However, the relatively low ROC-AUC values, ranging from approximately 0.61 to 0.63, indicate that the models have limited discriminative ability.

These results represent an exploratory predictive analysis rather than a production-ready credit risk system.

---

## 7. Credit Score Prediction — Regression

The second task was to predict each customer's numerical credit score.

Unlike classification, which predicts a default category, regression estimates a continuous value.

I evaluated three regression approaches:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### 7.1 Evaluation Metrics

Three metrics were used to compare predictive performance.

**Mean Absolute Error (MAE)**

Measures the average absolute difference between predicted and actual credit scores.

**Root Mean Squared Error (RMSE)**

Measures prediction error while penalizing larger errors more heavily.

**Coefficient of Determination (R²)**

Measures the proportion of variation in credit scores explained by the model on the test set.

### 7.2 Regression Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 21.64 | 29.68 | 0.7469 |
| Random Forest | 20.59 | 28.08 | 0.7735 |
| XGBoost | 21.68 | 29.53 | 0.7495 |

### 7.3 Regression Model Comparison

![Regression Model Comparison](/images/projects/credit_score/regression_model_compare.png)

**Random Forest achieved the strongest performance across all three metrics.**

Its R² of 0.7735 indicates that the model explained approximately 77.35% of the variation in credit scores on the test set.

The MAE of 20.59 means that predictions differed from actual credit scores by approximately 20.6 points on average.

### 7.4 Feature Importance

Predictive performance is only one part of credit risk analysis.

Understanding which financial variables influence predictions is equally important for interpreting model behavior.

I examined feature importance using the Random Forest Regressor.

![Features for predicting Credit Score](/images/projects/credit_score/features_predict_credit_score.png)

The visualization highlights the variables most influential in the Random Forest model.

Financial ratios and expenditure-related indicators contribute useful information to the prediction process.

These results support the importance of feature engineering, as relationships between financial variables can reveal information not immediately apparent from individual monetary values.

Feature importance describes the model's reliance on predictors rather than establishing a causal relationship with credit scores.

### Regression Insight

Random Forest's performance suggests that non-linear relationships and interactions between financial variables contribute useful predictive information.

However, Linear Regression also achieved a relatively strong R² of 0.7469.

This indicates that linear relationships explain a substantial portion of the variation in credit scores, while Random Forest provides an additional improvement.

---

## 8. Does PCA Improve Predictive Performance?

To evaluate the effectiveness of dimensionality reduction, I compared models trained on the original features with models trained using 18 principal components.

### 8.1 Classification with PCA

The PCA-transformed features were used to train Logistic Regression and XGBoost classifiers.

| Model | Accuracy | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 61.00% | 54.12% | 0.4402 | 0.6530 |
| XGBoost | 71.00% | 17.65% | 0.2564 | 0.5727 |

![Classification with PCA](/images/projects/credit_score/PCA_classification_model_compare.png)

Compared with the original feature space, PCA slightly improved Logistic Regression's recall and ROC-AUC.

However, its accuracy and F1-score decreased.

XGBoost also showed a reduction in ROC-AUC after PCA.

These results suggest that dimensionality reduction can affect different models in different ways.

### 8.2 Regression with PCA

The PCA-transformed features were also used to train Linear Regression and XGBoost regression models.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 27.08 | 37.04 | 0.6058 |
| XGBoost | 27.37 | 36.78 | 0.6113 |

![Regression with PCA](/images/projects/credit_score/PCA_regression_model_compare.png)

Both regression models performed substantially worse using PCA-transformed features.

For comparison, Linear Regression trained on the original features achieved an R² of 0.7469, while its PCA-based counterpart achieved 0.6058.

XGBoost also experienced a reduction in R² from approximately 0.75 to 0.61.

### Key Finding

**Preserving variance does not necessarily preserve predictive information.**

Although PCA retained approximately 90% of the dataset's variance, the resulting components did not consistently improve predictive performance.

Some lower-variance directions may contain information relevant to the target variables.

This demonstrates why dimensionality reduction should be evaluated based on downstream predictive performance rather than explained variance alone.

---

## 9. Additional Statistical Analysis

Beyond supervised machine learning, I explored statistical techniques to better understand relationships within the financial dataset.

### 9.1 Canonical Correlation Analysis (CCA)

Canonical Correlation Analysis was used to study the relationship between two groups of variables.

**Financial condition**

Income, savings, debt, and financial ratios.

**Spending behavior**

Expenditure across different spending categories.

The analysis investigated whether combinations of financial variables were associated with combinations of spending variables.

The first two canonical correlations were approximately:

| Canonical Component | Correlation |
|---|---:|
| First | 0.9914 |
| Second | 0.7538 |

These results suggest substantial shared structure between financial condition and spending behavior in the analyzed dataset.

However, a high canonical correlation does not establish causality or guarantee that one variable group can replace the other without losing predictive information.

### 9.2 Correspondence Analysis

Correspondence Analysis was also explored to investigate relationships between categorical financial attributes and default status.

The analysis examined associations involving credit score categories, gambling behavior, and default outcomes.

This provided an additional statistical perspective on how customer categories relate to observed financial outcomes.

Together, PCA, CCA, and Correspondence Analysis complemented the predictive models by exploring the underlying structure of the dataset.

---

## 10. Key Findings

The project produced several important findings.

**1. Credit risk classification requires more than accuracy.**

Logistic Regression achieved the highest recall for identifying defaulters, while Random Forest and XGBoost achieved higher overall accuracy but missed more defaulting customers.

**2. Random Forest performed best for credit score regression.**

The model achieved an R² of 0.7735, with an average absolute prediction error of approximately 20.6 credit score points.

**3. Financial ratios provide valuable predictive information.**

Ratios describing debt, savings, income, and expenditure help capture relationships between financial variables.

**4. Dimensionality reduction does not guarantee better predictions.**

PCA compressed the feature space into 18 components while retaining approximately 90% of variance, but regression performance deteriorated.

**5. Model choice depends on the prediction objective.**

Default detection and numerical credit score estimation require different evaluation criteria.

The results demonstrate the importance of selecting models based on the specific financial problem rather than relying on a single performance metric.

---

## 11. Technical Stack

| Category | Tools & Methods |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistical Analysis | PCA, CCA, Correspondence Analysis |
| Classification | Logistic Regression, Random Forest, XGBoost |
| Regression | Linear Regression, Random Forest, XGBoost |
| Machine Learning | Scikit-learn |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, MAE, RMSE, R² |

---

## Conclusion

This project demonstrates a structured machine learning workflow for credit risk analysis, combining statistical methods, financial feature engineering, predictive modeling, and model interpretation.

By investigating both default classification and credit score regression, I explored how different algorithms respond to the same financial dataset and why evaluation metrics must be chosen according to the prediction objective.

The analysis highlighted the importance of financial ratios, the trade-off between accuracy and minority-class recall, and the limitations of dimensionality reduction.

The most valuable lesson was understanding that a more complex model or a lower-dimensional feature space does not automatically produce better predictions.

**Effective credit risk modeling requires balancing predictive performance, interpretability, and practical financial considerations.**

---

## Limitations & Future Improvements

The results provide a useful foundation for exploratory credit risk modeling, but several improvements would be necessary before considering real-world deployment.

- Apply all learned preprocessing transformations and PCA within the training pipeline to prevent data leakage.
- Evaluate classification thresholds based on the relative costs of missed defaults and false alarms.
- Use cross-validation to assess model stability and tune hyperparameters.
- Investigate probability calibration and feature importance using additional interpretability techniques.
- Validate the models on an independent dataset to assess generalization.

These improvements would help establish more reliable performance estimates and strengthen the practical applicability of the models.