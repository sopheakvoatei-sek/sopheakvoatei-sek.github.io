---
title: "Credit Score Analysis & Risk Management | Visual & Technical Report"
date: "2025-12-31"
excerpt: "An illustrated credit-risk case study covering 1,000 customers, default classification, credit-score regression, financial feature engineering, PCA and model interpretation. Includes selected Python implementation examples and reported test results."
tags: ["Python", "Machine Learning", "Credit Risk", "Logistic Regression", "Random Forest", "XGBoost", "PCA", "Classification", "Regression"]
thumbnail: "/images/projects/credit_score/credit_score_distribution.png"
---

# Credit Score Analysis & Risk Management

**A visual and technical case study · ENSIIE · Modeling & Regularized Regression (MRR) · 2025–2026**

> **The central question:** Can financial behavior help us identify customers at risk of default *and* estimate a numerical credit score—and do the models that perform well at one task also perform well at the other?

| 1,000 | 84 | 18 | 0.7735 |
|:--|:--|:--|:--|
| Customers | Original features | PCA components at ~90.4% variance | Random Forest test R² for credit-score prediction |

![Credit score distribution](/images/projects/credit_score/credit_score_distribution.png)

*Figure 1 — Credit-score distribution from the project analysis.*

**At a glance.** I investigated two supervised learning problems on the same financial dataset: binary **DEFAULT** classification and continuous **CREDIT_SCORE** regression. The study combines exploratory analysis, engineered financial ratios, PCA, classification and regression models, and statistical methods for understanding relationships among variable groups.

> **Reading guide:** Each technical section follows **question → selected Python code → existing project figure → interpretation**. The numbers and images below come from the supplied report. The code blocks are **illustrative implementation patterns**, not recovered verbatim from the original notebook: the dataset file, exact column names, random seed, tuned hyperparameters, and original scripts were not supplied. Adapt the marked column names and rerun against the original data before claiming that a snippet reproduces a reported result.

---

## 01 / The problem and the data

### Two targets, two different definitions of success

| Task | Target | Output | Key evaluation question |
|:--|:--|:--|:--|
| Default detection | `DEFAULT` (0 = non-default; 1 = default) | Class / default probability | Are actual defaulters identified without too many false alarms? |
| Credit-score estimation | `CREDIT_SCORE` | Continuous score | How close are predicted scores to the observed ones? |

The dataset contains **1,000 customer records and 84 original financial and behavioral features**. According to the project report, these encompass income, debt and savings; financial ratios; spending across 11 categories; transaction history over 6- and 12-month periods; and financial attributes such as cards, mortgages and savings accounts.

### Implementation / first look at the dataset

```python
import pandas as pd

# Illustrative: replace with the original project file.
df = pd.read_csv("PATH_TO_ORIGINAL_DATA.csv")

print("Rows, columns:", df.shape)
print("Missing values:\n", df.isna().sum().sort_values(ascending=False).head(15))
print("Duplicate rows:", df.duplicated().sum())
print(df[["DEFAULT", "CREDIT_SCORE"]].describe(include="all"))
```

**What this check answers.** Before modeling, inspect missingness, duplicated customers, target types, and numeric ranges. Treat repeated customer records carefully: duplicate rows are not automatically errors if the data represents different observation periods.

### Workflow / from raw records to interpretation

```text
FINANCIAL + BEHAVIORAL RECORDS
             │
             ▼
Data audit → EDA → financial ratios → preprocessing
             │
             ├── Statistical structure: correlation / PCA / CCA / CA
             │
             ├── DEFAULT → classification → recall, F1, ROC-AUC
             │
             └── CREDIT_SCORE → regression → MAE, RMSE, R²
                                      │
                                      ▼
                           Interpretation + limitations
```

**Evaluation boundary.** The reported default-classification experiment uses a **70/30 stratified train/test split**, giving **300 test customers**. The supplied report does not specify every preprocessing or tuning detail, so the examples below demonstrate a leakage-conscious structure rather than asserting the exact historical execution order.

---

## 02 / Explore the targets before choosing a model

### 02.1 / What do the credit scores look like?

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.histplot(data=df, x="CREDIT_SCORE", hue="DEFAULT",
             bins=30, element="step", stat="count", ax=ax)
ax.set(title="Credit-score distribution by default status",
       xlabel="Credit score", ylabel="Customers")
fig.tight_layout()
plt.show()
```

![Credit score distribution by default status](/images/projects/credit_score/credit_score_distribution.png)

*Figure 2 — Examine score spread, overlap between target classes and possible extreme observations.*

**Interpretation.** A plot like this describes differences between observed groups, but it does not by itself demonstrate that a score can reliably predict default. The source report does not provide group means or a statistical test, so no numerical separation is asserted here.

### 02.2 / Is default a minority class?

```python
counts = df["DEFAULT"].value_counts().sort_index()
shares = df["DEFAULT"].value_counts(normalize=True).sort_index()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Non-default", "Default"], counts.reindex([0, 1], fill_value=0))
ax.set(title="Default class balance", ylabel="Customers")
for i, value in enumerate(counts.reindex([0, 1], fill_value=0)):
    ax.text(i, value, f"{value} ({shares.get(i, 0):.1%})",
            ha="center", va="bottom")
fig.tight_layout()
plt.show()
```

![Default class balance](/images/projects/credit_score/default_class_balance.png)

*Figure 3 — Class balance in the analyzed dataset.*

> **Why the class ratio matters:** Predicting “non-default” for most customers can yield respectable accuracy while missing many true defaulters. Read **recall, precision and F1** alongside accuracy; examine ROC-AUC for ranking discrimination.

| Metric | What it tells us |
|:--|:--|
| Accuracy | Fraction of all customers classified correctly |
| Precision | Of those flagged as defaulters, how many actually defaulted? |
| Recall | Of actual defaulters, how many were detected? |
| F1 | Balance between precision and recall |
| ROC-AUC | How well probabilities rank the two classes across thresholds |

---

## 03 / Make financial variables more informative

### 03.1 / Why use ratios, not just balances?

A debt of the same absolute amount can represent very different financial exposure depending on a customer's income and savings. Ratios express this context and may expose relationships obscured by raw monetary values.

| Feature idea | Calculation | What it expresses |
|:--|:--|:--|
| Debt-to-income | debt / income | Debt burden relative to income |
| Debt-to-savings | debt / savings | Debt relative to liquidity buffer |
| Savings-to-income | savings / income | Savings relative to income |
| Expenditure-to-income | expenditure / income | Spending relative to income |

### Implementation / example feature engineering

```python
import numpy as np

# EXAMPLE COLUMN MAP — replace keys with actual dataset field names.
cols = {
    "income": "INCOME",
    "debt": "DEBT",
    "savings": "SAVINGS",
    "spending": "TOTAL_EXPENDITURE",
}

# If a required column is absent, identify its true name rather than
# silently creating a substitute variable.
assert set(cols.values()).issubset(df.columns), "Update cols for your dataset"

def safe_ratio(numerator, denominator):
    # Zero/invalid denominators are missing, not meaningful zero ratios.
    return numerator.div(denominator.replace(0, np.nan))

engineered = df.copy()
engineered["debt_to_income"] = safe_ratio(engineered[cols["debt"]], engineered[cols["income"]])
engineered["debt_to_savings"] = safe_ratio(engineered[cols["debt"]], engineered[cols["savings"]])
engineered["savings_to_income"] = safe_ratio(engineered[cols["savings"]], engineered[cols["income"]])
engineered["expenditure_to_income"] = safe_ratio(engineered[cols["spending"]], engineered[cols["income"]])
```

**Interpretation.** Missing or zero denominators require an explicit policy. Ratios can become extreme when their denominators are near zero; inspect their distributions and choose any clipping or imputation using training data only. The supplied report identifies these ratio families but does not supply the exact underlying column names or missing-value treatment.

### 03.2 / How are predictors related?

```python
# Illustrative numeric correlation view; specify target and feature subset.
numeric = engineered.select_dtypes(include="number")
correlation = numeric.corr(method="pearson")
selected = correlation["CREDIT_SCORE"].drop("CREDIT_SCORE").abs()
top_features = selected.nlargest(12).index

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(correlation.loc[top_features, top_features],
            cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlations among selected numeric predictors")
fig.tight_layout()
plt.show()
```

![Feature correlation analysis](/images/projects/credit_score/feature_correlation.png)

*Figure 4 — Correlation analysis from the original project.*

**What the figure can and cannot tell us.** Financial ratios and spending-related measures were highlighted in the original analysis. Pearson correlation captures linear association, not causality or a definitive measure of a feature's contribution to a fitted model. Correlation-based screening also needs care to avoid using test-target information during feature selection.

---

## 04 / The experimental design

**A 70/30 split and separate preprocessing paths** help make the classification and regression analyses comparable. The example below keeps learned imputation, encoding and scaling inside a model pipeline. It is a recommended implementation template; the uploaded report does not provide the original pipeline code.

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = ["DEFAULT", "CREDIT_SCORE"]
X = engineered.drop(columns=TARGETS)
y_default = engineered["DEFAULT"]
y_score = engineered["CREDIT_SCORE"]

# Use one set of customer indices for both targets.
train_idx, test_idx = train_test_split(
    X.index, test_size=0.30, random_state=42,
    stratify=y_default
)
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_default_train, y_default_test = y_default.loc[train_idx], y_default.loc[test_idx]
y_score_train, y_score_test = y_score.loc[train_idx], y_score.loc[test_idx]

numeric_cols = X_train.select_dtypes(include="number").columns
categorical_cols = X_train.select_dtypes(exclude="number").columns

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])
```

> **Leakage checkpoint:** Split first; then fit preprocessing on training customers only. Exclude both outcome columns from predictors. If `CREDIT_SCORE` is calculated using later information, confirm its availability at the actual prediction time before considering it a legitimate predictor of `DEFAULT`.

---

## 05 / PCA: less information, or less *useful* information?

PCA seeks orthogonal directions of maximal predictor variance. It does **not** use the target to decide which directions matter for default or score prediction.

### Implementation / explained variance and component selection

```python
from sklearn.decomposition import PCA

# Dense conversion is suitable for this small illustrative dataset;
# the original implementation may have used a different representation.
X_train_ready = preprocess.fit_transform(X_train)
X_test_ready = preprocess.transform(X_test)
if hasattr(X_train_ready, "toarray"):
    X_train_ready = X_train_ready.toarray()
    X_test_ready = X_test_ready.toarray()

pca_full = PCA().fit(X_train_ready)
cumulative = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.searchsorted(cumulative, 0.90) + 1

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker=".")
ax.axhline(0.90, linestyle="--", label="90% target")
ax.axvline(n_components, linestyle=":", label=f"{n_components} components")
ax.set(xlabel="Number of components", ylabel="Cumulative explained variance",
       title="PCA explained variance")
ax.legend()
fig.tight_layout()
plt.show()
```

![Cumulative explained variance](/images/projects/credit_score/cumulative_explain_var.png)

*Figure 5 — The original analysis retained **18 principal components**, accounting for approximately **90.4%** of predictor variance.*

**Interpretation.** Compression is substantial, but a direction can have little overall variance and still be predictive of a target. We therefore compare actual test performance before calling PCA helpful. The component count shown in the figure is a reported project result, not guaranteed to emerge from the illustrative preprocessing above.

---

## 06 / Task A: predict default

### Candidate models

| Model | Modeling idea | Main consideration |
|:--|:--|:--|
| Logistic Regression | Linear decision boundary in transformed feature space | Coefficients can be examined, with caveats |
| Random Forest | Bagged trees capture interactions and nonlinearity | Default threshold can miss minority cases |
| XGBoost | Boosted trees learn successive corrections | Requires careful tuning and probability evaluation |

### Implementation / training templates

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, eval_metric="logloss", random_state=42
    ),
}

# These hyperparameters are EXAMPLES, not the original notebook settings.
classification_rows = []
for name, estimator in classifiers.items():
    model = Pipeline([("prep", preprocess), ("model", estimator)])
    model.fit(X_train, y_default_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    classification_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_default_test, predictions),
        "Precision": precision_score(y_default_test, predictions, zero_division=0),
        "Recall": recall_score(y_default_test, predictions, zero_division=0),
        "F1": f1_score(y_default_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_default_test, probabilities),
    })

example_classification_results = pd.DataFrame(classification_rows)
```

### Reported test results / 300 customers

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:--|--:|--:|--:|--:|--:|
| Logistic Regression | 66.67% | 42.72% | **51.76%** | **0.4681** | **0.6256** |
| Random Forest | 70.67% | 42.11% | 9.41% | 0.1538 | 0.6174 |
| XGBoost | **71.33%** | **48.65%** | 21.18% | 0.2951 | 0.6137 |

![Classification model comparison and ROC curves](/images/projects/credit_score/classification_model_compare.png)

*Figure 6 — Original classification comparison and ROC analysis.*

**How to read it.** Logistic Regression identified **51.76%** of the actual defaulters at the evaluated threshold, compared with **9.41%** for Random Forest and **21.18%** for XGBoost. XGBoost's higher overall accuracy does not mean it detects more defaulters. All three ROC-AUC values (~0.61–0.63) indicate limited discrimination in this experiment.

### Implementation / inspect ranking across thresholds

```python
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(7, 5))
for name, estimator in classifiers.items():
    model = Pipeline([("prep", preprocess), ("model", estimator)])
    model.fit(X_train, y_default_train)
    RocCurveDisplay.from_estimator(model, X_test, y_default_test,
                                  name=name, ax=ax)
ax.plot([0, 1], [0, 1], "--", label="Chance")
ax.set_title("ROC curves on the test set")
ax.legend()
fig.tight_layout()
plt.show()
```

### Which features influence the linear model?

```python
# Illustration: inspect a fitted Logistic Regression pipeline.
logit = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
logit.fit(X_train, y_default_train)

names = logit.named_steps["prep"].get_feature_names_out()
coefficients = pd.Series(logit.named_steps["model"].coef_[0], index=names)
print(coefficients.reindex(coefficients.abs().nlargest(15).index))
```

![Logistic regression feature coefficients](/images/projects/credit_score/logistic_reg_feature.png)

*Figure 7 — Coefficient view from the original Logistic Regression analysis.*

**Interpretation.** A positive coefficient raises the modeled log-odds of default when that feature increases, conditional on the other inputs. Magnitudes require attention to units, scaling and encoding; highly correlated predictors can make individual coefficients unstable. These are modeled associations, not causal effects.

> **Default-detection takeaway:** Which model is operationally useful depends on the cost of missed defaults, the burden of false alarms and the chosen decision threshold—not accuracy alone. The reported models are exploratory, not ready for lending decisions.

---

## 07 / Task B: estimate credit scores

A regression model predicts a **number**, not a default class. I compared Linear Regression, Random Forest and XGBoost on the numerical `CREDIT_SCORE` target.

### Implementation / fit and evaluate

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, random_state=42),
}

regression_rows = []
for name, estimator in regressors.items():
    model = Pipeline([("prep", preprocess), ("model", estimator)])
    model.fit(X_train, y_score_train)
    predicted = model.predict(X_test)
    regression_rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_score_test, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_score_test, predicted)),
        "R2": r2_score(y_score_test, predicted),
    })

example_regression_results = pd.DataFrame(regression_rows)
# These example settings need not reproduce the historical results below.
```

### Reported test results

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|:--|--:|--:|--:|
| Linear Regression | 21.64 | 29.68 | 0.7469 |
| **Random Forest** | **20.59** | **28.08** | **0.7735** |
| XGBoost | 21.68 | 29.53 | 0.7495 |

![Regression model comparison](/images/projects/credit_score/regression_model_compare.png)

*Figure 8 — Original credit-score regression comparison.*

**Interpretation.** Random Forest had the lowest reported MAE and RMSE and the highest reported test R². Its **MAE of 20.59** means an average absolute error of approximately **20.6 score points**; **R² = 0.7735** means it accounted for about 77.35% of the test-set variation relative to the mean-prediction baseline. Linear Regression's **R² = 0.7469** shows that a relatively simple relationship already explains substantial variation in this dataset.

### Which features does the forest use?

```python
forest = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
])
forest.fit(X_train, y_score_train)

feature_names = forest.named_steps["prep"].get_feature_names_out()
importance = pd.Series(forest.named_steps["model"].feature_importances_,
                       index=feature_names).nlargest(15)
importance.sort_values().plot.barh(figsize=(8, 6),
                                  title="Random Forest feature importance")
plt.tight_layout()
plt.show()
```

![Credit-score feature importance](/images/projects/credit_score/features_predict_credit_score.png)

*Figure 9 — Feature-importance results from the original regression analysis.*

**Interpretation.** Financial ratios and expenditure-related indicators appeared among the influential predictors in the source analysis. Standard tree impurity importance can favor features with many possible split points and distribute importance across correlated variables. A useful next step would be **permutation importance on held-out data**, with appropriate safeguards against leakage; importance is not proof of causality.

---

## 08 / PCA versus original features: the decisive comparison

### Implementation / a fair transformation comparison

```python
# Illustration: choose PCA using training predictors only.
from sklearn.base import clone

base_preprocessor = clone(preprocess)
Xtr = base_preprocessor.fit_transform(X_train)
Xte = base_preprocessor.transform(X_test)
if hasattr(Xtr, "toarray"):
    Xtr, Xte = Xtr.toarray(), Xte.toarray()

pca = PCA(n_components=0.90, svd_solver="full")
Xtr_pca = pca.fit_transform(Xtr)
Xte_pca = pca.transform(Xte)
print("Training-derived PCA components:", pca.n_components_)

# Fit each model on Xtr_pca, then evaluate on Xte_pca.
# Compare with the same model and evaluation procedure on original features.
```

**Important:** The code above is a leakage-aware demonstration. The supplied report states that 18 components captured ~90.4% variance, but does not establish whether the historic PCA was fitted only on training data. Do not treat these results as independently reproduced here.

### 08.1 / Classification after PCA

| Model | Accuracy | Recall | F1 | ROC-AUC |
|:--|--:|--:|--:|--:|
| Logistic Regression + PCA | 61.00% | **54.12%** | 0.4402 | **0.6530** |
| XGBoost + PCA | 71.00% | 17.65% | 0.2564 | 0.5727 |

![PCA classification model comparison](/images/projects/credit_score/PCA_classification_model_compare.png)

*Figure 10 — Original PCA-based classification comparison.*

**What changed?** Relative to its original-feature counterpart, Logistic Regression's recall increased from **51.76% to 54.12%** and ROC-AUC from **0.6256 to 0.6530**, while its F1 decreased. XGBoost's ROC-AUC fell from **0.6137 to 0.5727**. PCA therefore affected the two classifiers differently.

### 08.2 / Regression after PCA

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|:--|--:|--:|--:|
| Linear Regression + PCA | 27.08 | 37.04 | 0.6058 |
| XGBoost + PCA | 27.37 | 36.78 | 0.6113 |

![PCA regression model comparison](/images/projects/credit_score/PCA_regression_model_compare.png)

*Figure 11 — Original PCA-based regression comparison.*

**What changed?** Linear Regression fell from **R² = 0.7469** to **0.6058**; XGBoost fell from **0.7495** to **0.6113**. Retaining ~90% of predictor variance did not retain all information needed for these score predictions.

> **A useful lesson from the experiment:** Variance explained is a property of **X**, not a guarantee of information preserved about **y**. Assess dimensionality reduction on the downstream task, not on the variance threshold alone.

---

## 09 / Beyond prediction: relationships between variable groups

### 09.1 / Canonical Correlation Analysis (CCA)

CCA searches for linear combinations of two variable groups that correlate as strongly as possible. The source study compared **financial condition** (income, savings, debt and ratios) against **spending behavior** (expenditure categories).

| Canonical pair | Reported correlation |
|:--|--:|
| First | **0.9914** |
| Second | **0.7538** |

### Implementation / CCA pattern

```python
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# Replace these placeholders with the original exact feature lists.
financial_columns = ["INCOME", "SAVINGS", "DEBT"]
spending_columns = ["SPENDING_CATEGORY_1", "SPENDING_CATEGORY_2"]

# After confirming all columns and handling missing data appropriately:
# A = StandardScaler().fit_transform(df[financial_columns])
# B = StandardScaler().fit_transform(df[spending_columns])
# cca = CCA(n_components=2).fit(A, B)
# U, V = cca.transform(A, B)
# correlations = [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(2)]
```

**Interpretation.** The reported first canonical correlation is high within the analyzed sample, but in-sample CCA can overfit, especially with correlated or numerous variables. It does not demonstrate causality or prove that one group replaces the other. No original CCA figure was supplied, so none is presented as a project figure here.

### 09.2 / Correspondence Analysis (CA)

Correspondence Analysis explores associations in a **contingency table**. The source report states that it examined credit-score categories, gambling behavior and default outcomes, but provides neither the original table nor numerical CA coordinates.

```python
# Example of preparing a categorical contingency table.
# Update the placeholder names only after checking the dataset.
# contingency = pd.crosstab(
#     df["CREDIT_SCORE_CATEGORY"],
#     df["GAMBLING_BEHAVIOR"]
# )
# print(contingency)
# A CA implementation can then be applied to that table.
```

**Interpretation.** CA can help visualize category associations; it does not by itself establish which category causes default. The exact map and inertia cannot be reconstructed from the summary provided.

---

## 10 / Findings at a glance

| Observation | Evidence from this study | Practical meaning |
|:--|:--|:--|
| Accuracy can conceal missed defaults | Random Forest: 70.67% accuracy, 9.41% recall | Evaluate minority-class detection explicitly |
| Numeric scores were more predictable than default labels in these experiments | Random Forest score R² = 0.7735; classifier ROC-AUC ~0.61–0.63 | Different targets have different signal and evaluation needs |
| Engineered relationships matter | Ratios and spending variables highlighted in analysis | Contextualized financial features can be informative |
| PCA has trade-offs | 18 components, ~90.4% variance; lower score-regression R² | Evaluate transformations by downstream performance |
| Interpretation requires caution | Coefficients, tree importance and CCA associations | Association is not causation |

### Technical toolkit

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `XGBoost` · `PCA` · `CCA` · `Correspondence Analysis`

---

## 11 / What I would improve before real-world use

1. **Audit leakage and timing.** Ensure ratios, transformations, encoding and PCA are fitted on training data and use only information available at prediction time.
2. **Assess stability.** Repeat evaluation with cross-validation and an independent, later-time dataset where possible; the available results come from a limited sample.
3. **Tune classification decisions.** Compare decision thresholds using plausible costs for missed defaults and false alarms; inspect precision–recall behavior and probability calibration.
4. **Check subgroup behavior and data provenance.** Credit decisions can carry serious consequences. Assess fairness, regulatory requirements, consent, and whether the data represents the intended lending population.
5. **Make interpretation more robust.** Compare permutation importance and error analysis across subgroups, and inspect large prediction errors rather than relying on one global score.

> **Project conclusion:** More complex models and fewer dimensions are not automatically better. In this dataset, model evaluation changed meaningfully when the target shifted from default detection to score estimation—and when PCA replaced the original feature space. The strongest lesson is to choose the modeling and evaluation workflow around the decision the model is meant to support.

---

### Reproducibility note

**Reported results and image paths** are preserved from the supplied project report. **Python examples are newly authored explanatory snippets**, not verified extracts from the project notebook. Dataset paths, feature names, categorical levels, exact hyperparameters and some statistical-analysis inputs must be replaced with the original project values. Until that source code and dataset are provided, the snippets should not be represented as reproducing the tables or figures exactly.