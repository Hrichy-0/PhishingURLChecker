"""Project paths and constants shared by scripts, tests, and the API."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "phishguard_model.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

KAGGLE_DATASET = "joebeachcapital/phiusiil-phishing-url"

# We use an intuitive internal convention even though PhiUSIIL uses the reverse.
LEGITIMATE_LABEL = 0
PHISHING_LABEL = 1
