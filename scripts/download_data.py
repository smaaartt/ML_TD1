import hashlib
import json
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

dest = Path("data/raw/telco.csv")
meta = Path("data/raw/telco.source.json")
dest.parent.mkdir(parents=True, exist_ok=True)

if not dest.exists():
    urllib.request.urlretrieve(URL, dest)

actual = hashlib.sha256(dest.read_bytes()).hexdigest()
if actual != EXPECTED_SHA256:
    raise ValueError("Source hash mismatch: investigate before continuing.")

meta.write_text(
    json.dumps(
        {"source_url": URL, "sha256": actual, "file": dest.as_posix()},
        indent=2,
    ),
    encoding="utf-8",
)
print(f"OK: {dest} ({actual})")