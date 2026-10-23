import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from churnlab.data import (
    FEATURES, SEED, TARGET, encode_target, load_raw, parse_inputs, split_data,
)
from churnlab.evaluate import evaluate_model, timed_fit
from churnlab.pipeline import build_pipeline

PRIMARY_METRIC = "average_precision"  # choisie AVANT la comparaison

df = parse_inputs(load_raw())
train, valid, test = split_data(df)  # test : réservé, jamais utilisé ici

X_train, y_train = train[FEATURES], encode_target(train[TARGET])
X_valid, y_valid = valid[FEATURES], encode_target(valid[TARGET])

models = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic": LogisticRegression(max_iter=1000, random_state=SEED),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1
    ),
}

results = []
for name, classifier in models.items():
    candidate = build_pipeline(classifier)  # instance neuve à chaque fois
    seconds = timed_fit(candidate, X_train, y_train)
    metrics = evaluate_model(candidate, X_valid, y_valid)
    results.append({
        "model": name,
        "params": {k: str(v) for k, v in classifier.get_params().items()},
        "fit_seconds": round(seconds, 3),
        **metrics,
    })

learned = [r for r in results if r["model"] != "baseline"]
best = max(learned, key=lambda r: r[PRIMARY_METRIC])

out = Path("evidence/tp02")
out.mkdir(parents=True, exist_ok=True)
(out / "comparison.json").write_text(
    json.dumps({"primary_metric": PRIMARY_METRIC, "selected": best["model"],
                "results": results}, indent=2),
    encoding="utf-8",
)

for r in results:
    print(f'{r["model"]:14} AP={r["average_precision"]:.3f} '
          f'P={r["precision"]:.3f} R={r["recall"]:.3f} F1={r["f1"]:.3f} '
          f'acc={r["accuracy"]:.3f} fit={r["fit_seconds"]}s')
print("Sélectionné :", best["model"])


import hashlib
import platform
import subprocess
import sys
from importlib.metadata import version

import joblib

from churnlab.data import CATEGORICAL, EXPECTED_SHA256, NUMERIC
from churnlab.evaluate import THRESHOLD

selected_name = best["model"]
selected_pipeline = build_pipeline(models[selected_name].__class__(
    **models[selected_name].get_params()
))
selected_pipeline.fit(X_train, y_train)

art = Path("evidence/tp02")
joblib.dump(selected_pipeline, art / "pipeline.joblib")
artifact_sha = hashlib.sha256((art / "pipeline.joblib").read_bytes()).hexdigest()

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"

metadata = {
    "selected_model": selected_name,
    "positive_class": "Churn = Yes (1)",
    "threshold": THRESHOLD,
    "primary_metric": PRIMARY_METRIC,
    "validation_metrics": best,
    "source_sha256": EXPECTED_SHA256,
    "artifact_sha256": artifact_sha,
    "split": {
        "seed": SEED, "ratios": "60/20/20 stratified",
        "train_ids": train["customerID"].tolist(),
        "valid_ids": valid["customerID"].tolist(),
        "test_ids": test["customerID"].tolist(),
    },
    "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
    "preprocessing": {
        "numeric": "median imputation + StandardScaler",
        "categorical": "most_frequent imputation + OneHotEncoder(handle_unknown=ignore)",
    },
    "model_params": best["params"],
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": commit,
        "packages": {n: version(n) for n in
                     ["scikit-learn", "pandas", "numpy", "joblib"]},
    },
}
(art / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# Échantillon de référence pour comparer rechargement et original
sample = valid[FEATURES].head(20)
reference = selected_pipeline.predict_proba(sample)[:, 1]
sample.to_csv(art / "sample_inputs.csv", index=False)
(art / "reference_scores.json").write_text(
    json.dumps(reference.tolist()), encoding="utf-8")
print("Artefact sauvegardé :", art / "pipeline.joblib")
