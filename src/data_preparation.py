"""Loading, cleaning, and X/y construction for the Telco churn dataset.

Everything here is deterministic and rule-based (no statistics are learned
from the data), so it is safe to run BEFORE the train/test split.
Anything that learns from data (imputation statistics, scaling, encoding
categories) lives in `preprocessing.py` and is fitted on training data only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_COL, POSITIVE_LABEL, PROCESSED_PATH, RAW_PATH, TARGET

# Columns that would exist only after (or because of) the churn event.
# They are absent from the common 21-column CSV but present in some
# expanded versions of the dataset. Using them would be target leakage.
LEAKAGE_COLUMNS = ["Churn Label", "Churn Value", "Churn Score", "Churn Reason"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as provided (no modifications)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Telco Customer Churn CSV "
            "and place it in data/raw/ (see README.md)."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply rule-based cleaning. Returns (clean_df, log of what was changed).

    Decisions (each is logged so it can be reported in the notebook):
      1. Strip whitespace in text columns (blank strings become missing).
      2. Convert TotalCharges to numeric; unparsable values become NaN.
      3. TotalCharges missing AND tenure == 0 -> 0. A customer with zero months
         of tenure has not been billed yet, so 0 is a domain-based value.
         Any other missing TotalCharges is left as NaN and imputed later,
         inside the pipeline, using training data only.
      4. SeniorCitizen 0/1 -> "No"/"Yes" so all binary flags are handled the same.
      5. Drop exact duplicate rows (all columns, including customerID).
      6. Drop rows with a missing/invalid target (cannot be used for supervised learning).
    """
    log: dict = {"rows_in": len(df)}
    df = df.copy()

    # 1. whitespace
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            stripped = df[col].astype("string").str.strip()
            keep = (stripped.notna() & (stripped != "")).fillna(False).astype(bool)
            df[col] = stripped.astype(object).where(keep, np.nan)
    log["missing_before_conversion"] = int(df.isna().sum().sum())

    # 2. TotalCharges -> numeric
    if "TotalCharges" in df.columns:
        before_na = int(df["TotalCharges"].isna().sum())  # blank strings already became NaN in step 1
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        after_na = int(df["TotalCharges"].isna().sum())
        log["totalcharges_blank_or_missing_in_raw"] = before_na
        log["totalcharges_other_non_numeric_coerced"] = after_na - before_na

        # 3. domain rule for brand-new customers
        new_customer = df["TotalCharges"].isna() & (df["tenure"] == 0)
        log["totalcharges_filled_with_0_tenure0"] = int(new_customer.sum())
        df.loc[new_customer, "TotalCharges"] = 0.0
        log["totalcharges_missing_left_for_pipeline"] = int(df["TotalCharges"].isna().sum())

    # 4. SeniorCitizen
    if "SeniorCitizen" in df.columns and pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Validity checks (reported, not silently fixed)
    numeric_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    log["negative_numeric_values"] = int((df[numeric_cols] < 0).sum().sum())

    # 5. duplicates
    n_dupes = int(df.duplicated().sum())
    log["duplicate_rows_dropped"] = n_dupes
    df = df.drop_duplicates().reset_index(drop=True)

    # 6. target validity
    valid_target = df[TARGET].isin(["Yes", "No"])
    log["rows_dropped_invalid_target"] = int((~valid_target).sum())
    df = df.loc[valid_target].reset_index(drop=True)

    log["rows_out"] = len(df)
    return df, log


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into features X and binary target y (1 = churn)."""
    leaking = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if leaking:
        print(f"WARNING: dropping leakage columns: {leaking}")

    y = (df[TARGET] == POSITIVE_LABEL).astype(int).rename("churn")
    X = df.drop(columns=[TARGET, ID_COL, *leaking], errors="ignore")
    return X, y


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
