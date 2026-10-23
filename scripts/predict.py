import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnlab.data import FEATURES, parse_inputs

art = Path("evidence/tp02")
pipeline = joblib.load(art / "pipeline.joblib")  #artefact de confiance 

new_customers = parse_inputs(pd.read_csv(art / "sample_inputs.csv"))[FEATURES]
scores = pipeline.predict_proba(new_customers)[:, 1]
reference = np.array(json.loads((art / "reference_scores.json").read_text()))

TOLERANCE = 1e-9
print("écart max :", float(np.abs(scores - reference).max()))
assert np.allclose(scores, reference, atol=TOLERANCE), "Prédictions différentes !"
print(f"OK : scores identiques à l'original (tolérance {TOLERANCE})")