# TP02 - Answers

## 1. First model and baseline

Task: predict `Churn = Yes` (positive class = 1). `customerID` and `Churn` are never used as input features. Split: stratified 60/20/20, seed 42. The test set is reserved and was never used.

Validation set: 1,409 customers (1,035 "No", 374 "Yes"). Matrices use labels [0 = No, 1 = Yes]; rows = actual, columns = predicted.

| Model (features: tenure, MonthlyCharges) | Accuracy | Confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---|
| Majority baseline (fitted on train) | 0.7346 | [[1035, 0], [374, 0]] |
| Logistic regression | 0.7828 | [[950, 85], [221, 153]] |

**What would a system that always predicts "No churn" achieve?**
About 73.5% accuracy on validation, with no churner detected.

**Can high accuracy coexist with failing to identify any churners?**
Yes. The baseline has 73.5% accuracy but recall 0, because most customers do not churn. Accuracy alone is misleading on imbalanced data.

**Which data are permitted to influence model selection?**
Training data is used to fit models. Validation data is used to compare and choose them. The test set must stay untouched until the final evaluation.

## 2. Preprocessing pipeline

Features: numeric `tenure`, `MonthlyCharges`, `TotalCharges`; categorical `Contract`, `InternetService`, `PaymentMethod`.

Documented policy:
- Blank `TotalCharges` (11 rows) becomes missing. Non-blank non-numeric text raises an error (`pd.to_numeric(errors="raise")`), so malformed data is never silently accepted. The original CSV is never modified.
- Numeric branch: median imputation, then `StandardScaler`.
- Categorical branch: most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")`.
- Unknown category at inference: encoded as all zeros. This keeps prediction working but does not make the prediction reliable for that category.
- All statistics are learned on training data only.

Result of one `fit` call with logistic regression (6 features): accuracy 0.8041, confusion matrix [[935, 100], [176, 198]]. This cannot be compared directly to milestone 1 as proof of a better algorithm, because the features also changed (2 to 6).

**What do `fit`, `transform` and `predict` each do?**
- Fit, transform & predict : 
  - fit learn paramters from the training like median, mean, ecart type , coefficient etc
  - transform => will apply what it learned already 
  - predict => preprocessing and then will classsify 

**What goes wrong if you fit a scaler on validation data?**
If we fit a scaler on validation data, the meann ecart type etc will be calculated from validation data which normally should'nt affect the model, it's data leakage 

**Does using a Pipeline automatically prevent leakage if you give it the wrong training data?**
If we fit a scaler on validation data, the meann ecart type etc will be calculated from validation data which normally should'nt affect the model, it's data leakage 

**Which operations are ordered by `Pipeline`, and which columns are routed by `ColumnTransformer`?**
the Pipeline will assure that the preprocessing is applied on the fitted data & ColumnTransformer  will route the columns through the right branch (category & numerical)

## 3. Fair model comparison

Same split, same 6 features, same metric definitions, a fresh pipeline for each candidate. Threshold 0.5 for precision, recall and F1.

| Model | Avg. precision | Precision | Recall | F1 | Accuracy | Fit time |
|---|---|---|---|---|---|---|
| Majority baseline | 0.265 | 0.000 | 0.000 | 0.000 | 0.735 | 0.02 s |
| Logistic regression | 0.647 | 0.664 | 0.529 | 0.589 | 0.804 | 0.04 s |
| Decision tree (max_depth=5) | 0.603 | 0.630 | 0.602 | 0.616 | 0.801 | 0.02 s |
| Random forest (200 trees, max_depth=8) | **0.670** | 0.661 | 0.505 | 0.573 | 0.800 | 0.48 s |

Full machine-readable table: `evidence/tp02/comparison.json`. Fit times are descriptive and specific to my machine.

**Selection (see also `selection.md`).**
Primary metric chosen before the comparison: average precision on positive-class scores, because classes are imbalanced (26.5% churn) and it does not depend on a threshold. The decision uses validation only. I keep the **random forest**: it has the best average precision (0.670), and I applied the rule fixed in advance. The gap with logistic regression (+0.023) is small and comes from a single split, so I do not claim a proven improvement. The extra cost is negligible (about 0.5 s).

**Would changing the split for every classifier make the comparison easier to interpret?**
No. Different splits would mix the effect of the model with the effect of the data, so differences could not be attributed to the algorithm.

**Does the model with the highest accuracy necessarily meet the team's needs?**
No. Accuracy is almost identical for all three models (about 0.80), and it hides weak recall. For churn, finding churners matters, so recall, precision and average precision are more informative.

**If two candidates are close, what would justify choosing the simpler one?**
Simplicity, speed, easier interpretation and maintenance, and the fact that a small gap on a single split may be noise.

**Is a predicted probability of 0.8 automatically a well-calibrated 80% risk?**
No. Models can rank customers well while their probabilities are over- or under-confident. Calibration must be checked separately.

## 4. Reusable code and saved artifact

Structure: `data.py` (loading with hash check, parsing, split), `pipeline.py` (builds an unfitted pipeline), `evaluate.py` (metrics, timing), `scripts/train.py` (terminal entry point), `scripts/predict.py` (separate inference process). `build_pipeline` returns an unfitted object and does not read data or train.

Saved in `evidence/tp02/`: `pipeline.joblib` (fitted preprocessing + random forest, about 4.3 MB), `metadata.json` (source hash, artifact hash, seed, customer IDs of each split, features, preprocessing settings, model parameters, metrics, positive class, threshold, Python/library versions, git commit).

Reload check: `scripts/predict.py` loads the artifact in a new Python process, without calling `fit` and without labels. Scores for 20 validation customers match the original within a tolerance of 1e-9 (maximum difference 1.1e-16).

**Why is saving only the classifier insufficient?**
At inference, raw columns must go through the same imputation, scaling and encoding as in training. A classifier alone expects the transformed array and would fail or give wrong results.

**Which fitted information lives inside an imputer or encoder?**
The imputer stores the learned fill values (`statistics_`: medians, most frequent categories). The scaler stores means and standard deviations. The encoder stores the list of known categories.

**Could two files called `model.joblib` contain different models?**
Yes. A filename is not an identity. That is why the metadata records the artifact SHA-256, the source data hash, parameters, versions and git commit.

**How does a sklearn Pipeline differ from a workflow DAG such as Kubeflow Pipelines?**
A sklearn Pipeline chains estimators inside one Python object and process (fit, transform, predict). A workflow DAG orchestrates separate containerized steps with dependencies, scheduling, retries and artifacts passed between steps, possibly on a cluster.

## 5. Reliability checks (`tests/test_checks.py`, 6 tests passed)

| Check | Evidence |
|---|---|
| No customer overlap between splits | `test_no_customer_overlap`: train, valid and test IDs are disjoint and total 7,043 |
| Imputation statistics come from training data | `test_imputer_uses_train_only`: `statistics_` equal the training-only medians |
| Prediction does not refit | `test_predict_does_not_change_statistics`: `statistics_` unchanged after `predict` |
| Missing numeric value and unknown category follow the policy | `test_missing_numeric_and_unknown_category`: missing `TotalCharges` becomes the median, an unseen `Contract` becomes zeros, no NaN in output |
| Missing required feature fails visibly | `test_missing_required_feature_fails`: dropping `Contract` raises an error |
| Save/reload preserves predictions | `scripts/predict.py`: max difference 1.1e-16, no `fit`, no labels |

Handling an unknown category does not prove that predictions are reliable for it.

## Peer review

- solo

## Limitations and remaining work

- One validation split: small differences between models may be noise (cross-validation would help).
- Recall is about 50% at threshold 0.5; no threshold policy yet.
- Probabilities are not calibrated.
- The test set is reserved and not yet used; a final evaluation should be done once, after selection is frozen.
