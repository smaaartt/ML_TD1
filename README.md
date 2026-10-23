## TP02: training pipeline

Run from the project root:

```sh
uv sync --locked
uv run python scripts/download_data.py
uv run python scripts/train.py      # compares 3 models + baseline, saves the selected pipeline
uv run python scripts/predict.py    # reloads the artifact in a new process, no fit
uv run python -m pytest -v          # reliability checks
```

- Task: predict `Churn = Yes` (positive class). Features: tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod. `customerID` and `Churn` are never inputs.
- Split: stratified 60/20/20, seed 42. The test set is reserved and unused.
- Selection: average precision on validation, chosen before the comparison (see `evidence/tp02/selection.md`).
- Outputs in `evidence/tp02/`: `comparison.json`, `pipeline.joblib`, `metadata.json`, `selection.md`.
- Only load `pipeline.joblib` if you trust it, and use the library versions listed in `metadata.json`.

### Limitations
- One validation split: small differences between models may be noise.
- Recall is about 50% at threshold 0.5; no threshold tuning yet.
- Probabilities are not guaranteed to be calibrated.
- Unknown categories are encoded as zeros: this does not make predictions reliable for them.
- Tracking and model versioning come in the next TP.