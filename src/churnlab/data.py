import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/telco.csv")
EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Churn"
SEED = 42


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Lit le CSV après vérification du SHA-256. Ne modifie pas le fichier."""
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError("Source hash mismatch: investigate before continuing.")
    return pd.read_csv(path)


def parse_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Parsing sans état : TotalCharges vide -> NaN, autre texte -> erreur."""
    out = df.copy()
    total = out["TotalCharges"].astype("string").str.strip()
    total = total.mask(total == "")
    out["TotalCharges"] = pd.to_numeric(total, errors="raise")
    return out


def encode_target(series: pd.Series) -> pd.Series:
    """Churn = Yes est la classe positive (1)."""
    y = series.map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected Churn labels.")
    return y.astype(int)


def split_data(df: pd.DataFrame, seed: int = SEED):
    """Split stratifié 60/20/20 : retourne train, valid, test."""
    train, rest = train_test_split(
        df, test_size=0.4, stratify=df[TARGET], random_state=seed
    )
    valid, test = train_test_split(
        rest, test_size=0.5, stratify=rest[TARGET], random_state=seed
    )
    return train, valid, test