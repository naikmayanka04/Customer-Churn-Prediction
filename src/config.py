from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "baseline_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "Churn"          # "Yes" / "No" in the raw file
ID_COL = "customerID"     # identifier, carries no predictive signal
POSITIVE_LABEL = "Yes"    # churn = positive class (encoded as 1)
