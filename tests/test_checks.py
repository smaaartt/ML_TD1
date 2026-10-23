import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from churnlab.data import (
    FEATURES, NUMERIC, TARGET, encode_target, load_raw, parse_inputs, split_data,
)
from churnlab.pipeline import build_pipeline


@pytest.fixture(scope="module")
def data():
    df = parse_inputs(load_raw())
    return split_data(df)


@pytest.fixture(scope="module")
def fitted(data):
    train, _, _ = data
    pipe = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=42))
    pipe.fit(train[FEATURES], encode_target(train[TARGET]))
    return pipe


def test_no_customer_overlap(data):
    train, valid, test = data
    ids = [set(d["customerID"]) for d in (train, valid, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == 7043


def test_imputer_uses_train_only(data, fitted):
    train, _, _ = data
    stats = fitted.named_steps["preprocessor"].named_transformers_["num"] \
        .named_steps["imputer"].statistics_
    expected = train[NUMERIC].median().to_numpy()
    assert np.allclose(stats, expected)


def test_predict_does_not_change_statistics(data, fitted):
    _, valid, _ = data
    imp = fitted.named_steps["preprocessor"].named_transformers_["num"] \
        .named_steps["imputer"]
    before = imp.statistics_.copy()
    fitted.predict(valid[FEATURES])
    assert np.array_equal(before, imp.statistics_)


def test_missing_numeric_and_unknown_category(data, fitted):
    _, valid, _ = data
    batch = valid[FEATURES].head(5).copy()
    batch.loc[batch.index[0], "TotalCharges"] = np.nan
    batch.loc[batch.index[1], "Contract"] = "Contrat inconnu"
    proba = fitted.predict_proba(batch)
    assert proba.shape == (5, 2) and not np.isnan(proba).any()


def test_missing_required_feature_fails(data, fitted):
    _, valid, _ = data
    batch = valid[FEATURES].head(5).drop(columns=["Contract"])
    with pytest.raises(Exception):
        fitted.predict(batch)