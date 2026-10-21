# TP01 - Réponses

## Partie A

**Question A1:** Why should the original CSV, a fitted model and the training source code have different roles in the repository?


the csv is the entrance data and it should not be modified
the code produce results
the modele is the generated output from code & data so it should not be versioned like source code

**A2.** Why would including `Churn` among the model inputs invalidate the prediction task?

Churn is the measure that we want to predict so if we uses as input to the model it will have a target leakage and not predict but read data only

## Partie B2

- **B2.1** `requires-python = ">=3.11"`; direct dependency: `pandas>=3.0.6`
- **B2.2** In `uv.lock`: pandas version 3.0.6
- **B2.3** Transitive dependency: `python-dateutil` v2.9.0.post0 (required by pandas), which itself depends on `six`
- **B2.4** `pyproject.toml` = requested constraints; `uv.lock` = exact resolved versions (including transitive ones); `.venv/` = locally installed environment, generated and not committed

## Part C3


1. Classification: we predict a category (Yes/No), not a continuous value.
2. 26.54% churn means the classes are imbalanced. A model that always answers "No" would reach about 73% accuracy while detecting nothing, so accuracy alone is misleading (use recall, precision, F1, AUC).
3. `TotalCharges` contains 11 blank values (" "), so pandas reads it as text (`str`) rather than as a number.
4. `isna()` detects NaN values, but " " is a non-empty string: it must be tested with `strip() == ""`.
5. The SHA-256 hash (and the commit-pinned URL): if the hash is identical, the file bytes are identical.


## Partie D

After renaming `.venv`, `uv sync --locked` recreated the environment (20 packages resolved, `uv.lock` unchanged). The before/after records are identical: Python 3.11.15, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, churnlab 0.1.0. The CSV hash and the inspection counts are identical.

| Item | Evidence produced in TP01 | Remaining limitation |
|---|---|---|
| Code | Git commit | Uncommitted edits are not identified by that commit |
| Package selections | `pyproject.toml`, `uv.lock` | Platform/interpreter selection can affect installations |
| Installed environment | Before/after environment records | This does not capture every OS or hardware detail |
| Source data | Source URL and SHA-256 | This does not establish label quality or absence of leakage |
| Trained model | Not produced in TP01 | Requires later training/configuration/artifact evidence |


## Par E

- **URL PR:** [https://github.com/smaaartt/ML_TD1/pull/1]
- **Hash du commit (fusion) :** [sortie de `git log --oneline`]
- **Question asked by my partner and my answer:** did solo 

## Part F


- **F1** (`git restore README.md`): discards the edit to the file; the change is lost.
- **F2** (`git restore --staged README.md`): only removes the file from the staging area; the edit stays in the working file.

## Ce que TP01 ne reproduit pas encore

Training results (no model yet), random seeds and training configuration, hardware and OS details, label quality, and absence of data leakage.