# Predicting Confidence in Social Institutions: A Classification Analysis Using World Values Survey Data

Individual classification analysis comparing five traditional classifiers and an Artificial Neural Network on World Values Survey (WVS) data, predicting binary confidence (Low / High) in three social institutions: the civil service, churches, and the armed forces.

## Overview

- **Dataset:** WVS binary extract, randomly sampled to 20,000 observations with 30 predictor variables and 3 binary class variables (`CCivilService`, `CChurches`, `CArmedForces`)
- **Pre-processing:** Negative-coded non-substantive survey responses recoded to NA, complete-case filtering (20,000 → 7,952 observations), categorical/ordinal variables converted to factors
- **Split:** 70:30 train/test (5,566 train / 2,386 test), `set.seed(34091904)`
- **Models implemented:** Decision Tree (`rpart`/`tree`), Naive Bayes (`e1071`), Bagging (`adabag`/`ipred`), Boosting (`adabag`), Random Forest (`randomForest`), Artificial Neural Network (`nnet`)
- **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC/AUC — F1 and AUC prioritised over accuracy due to class imbalance in `CChurches` and `CArmedForces`

## Key Findings

**Overall best model across all three class variables: Random Forest**
- Average F1-score: 0.705, Average AUC: 0.681, Average Recall: 0.759 (across CCivilService, CChurches, CArmedForces)

**Per-institution results:**

| Class Variable | Best Model | F1-Score | AUC |
|---|---|---|---|
| CCivilService | Boosting | 0.5326 | 0.6220 |
| CChurches | Random Forest | 0.8191 | 0.8128 |
| CArmedForces | Bagging | 0.7846 | 0.6211* |

\* Bagging had the highest F1/recall for CArmedForces but low precision (0.6533) from over-predicting the majority class; Random Forest offered the more balanced trade-off (F1 = 0.7745, AUC = 0.6211, the highest AUC for this variable).

**Model improvement:** The weakest ensemble model (Bagging on CCivilService, original F1 = 0.4905) was retuned via 3-fold cross-validation across predictor sets and hyperparameters (`nbagg`, `cp`, `minsplit`, `maxdepth`), then had its classification threshold adjusted from 0.50 to 0.30. This raised F1-score to 0.6104 and AUC from 0.5889 to 0.5999, at the cost of accuracy (0.5733 → 0.5109) — an accepted trade-off given the class imbalance.

**ANN classifier:** A single-hidden-layer ANN (`nnet`, tuned over hidden layer size, decay, and iterations via internal 80:20 validation split) was trained on South Africa (ZAF) data — the country with the most observations post-cleaning — to predict `CCivilService`. It achieved F1 = 0.5902 and AUC = 0.6476, the highest of any model (including the five classifiers above) for that class variable. Performance was then compared across the two waves with the most test observations: Wave 3 (F1 = 0.6154, AUC = 0.6615) outperformed Wave 6 (F1 = 0.5714, AUC = 0.6201), suggesting some temporal variation in predictor-outcome relationships.

**Attribute importance:** `PolScale`, `PrivateState`, `IncomeEquality`, `ILPolitics`, `PolInterest`, and `ILReligion` were consistently important predictors across all three institutions. `ILReligion` dominated prediction of confidence in churches; political and army-related attitudes dominated confidence in the armed forces; civil service confidence relied on a broader mix of political, state, and economic predictors rather than one dominant variable.

## Repository Structure

- `r_code/`
  - `00_libraries.R` — Required R packages
  - `01_data_exploration.R` — Predictor distributions, class balance
  - `02_data_preprocessing.R` — Missing value / negative-code handling, complete-case filtering
  - `03_train_test_split.R` — 70:30 train/test split
  - `04_classification_models.R` — Decision Tree, Naive Bayes, Bagging, Boosting, Random Forest
  - `05_performance_metrics.R` — Accuracy/Precision/Recall/F1 calculation
  - `06_roc_auc.R` — ROC curves and AUC computation
  - `07_model_comparison.R` — Combined metrics + AUC comparison across models
  - `08_attribute_importance.R` — Variable importance extraction and plots
  - `09_model_improvement_bagging.R` — CV-tuned + threshold-adjusted improved Bagging model
  - `10_ann_classifier.R` — ANN implementation and tuning (ZAF, CCivilService)
  - `11_ann_wave_comparison.R` — ANN performance across survey waves
- `full_analysis.R` — All scripts combined in execution order

## Tools

R, `rpart`, `tree`, `e1071`, `adabag`, `ipred`, `randomForest`, `nnet`, `pROC`, `ggplot2`, `dplyr`, `tidyr`, `corrplot`

## Note

This code is reconstructed from the appendix of the original assignment report (submitted as a PDF) rather than from a live `.R` project file, and has been reorganised into separate scripts by analysis stage for readability. Logic and parameters are unchanged from the original.
