"""
data_loader.py

Reusable data loading functions for the Dual Credit Risk Assessment Platform.

Design principle: every function here reads ONLY from data/raw/ and never
modifies it. This keeps raw data immutable, which matters for audit and
reproducibility (see Module 2 / Module 4 notes).
"""

import pandas as pd
from pathlib import Path

# Project root is assumed to be two levels up from this file (src/data_loader.py
# -> project root). Adjust here once, and every function below stays correct.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_individual_data(sample: bool = True) -> pd.DataFrame:
    """
    Load the LendingClub (individual borrower) dataset.

    Parameters
    ----------
    sample : bool, default True
        If True, loads the 10,000-row stratified sample (loan_sample.csv) —
        use this while developing and testing notebook logic.
        If False, loads the full raw file (loan.csv, ~1.1GB) — use this only
        for the final full-scale training run on your own machine.

    Returns
    -------
    pd.DataFrame
    """
    filename = "loan_sample.csv" if sample else "loan.csv"
    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find {filepath}. Confirm the file is in data/raw/."
        )

    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded individual data: {filename} -> shape {df.shape}")
    return df


def load_sme_data() -> pd.DataFrame:
    """
    Load the Corporate Credit Rating (SME borrower) dataset.

    Returns
    -------
    pd.DataFrame
    """
    filepath = RAW_DATA_DIR / "corporateCreditRatingWithFinancialRatios.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find {filepath}. Confirm the file is in data/raw/."
        )

    df = pd.read_csv(filepath)
    print(f"Loaded SME data: shape {df.shape}")
    return df


if __name__ == "__main__":
    # Quick manual test: run `python data_loader.py` from the src/ folder
    # to confirm both loaders work before importing them into a notebook.
    individual_df = load_individual_data(sample=True)
    sme_df = load_sme_data()

    print("\nIndividual data preview:")
    print(individual_df.head(3))

    print("\nSME data preview:")
    print(sme_df.head(3))
