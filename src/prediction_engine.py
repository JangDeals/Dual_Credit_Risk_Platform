"""
prediction_engine.py

Reusable scoring functions for the Dual Credit Risk Assessment Platform.

Each function:
1. Loads the versioned model package (model + preprocessor + thresholds)
2. Accepts a new applicant's raw feature values as a dictionary
3. Runs them through the exact same preprocessing pipeline used in training
4. Returns a structured result: PD, risk grade, and a recommendation

This is the only file a Streamlit app (or anything else) needs to import
to score a new applicant -- callers never touch the model internals directly.
"""

import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _assign_grade(probability, edges, labels):
    for i in range(len(labels)):
        if edges[i] <= probability <= edges[i + 1]:
            return labels[i]
    return labels[-1]


def _recommendation(grade):
    """Simple, transparent recommendation logic based on risk grade tier.
    A real bank would tune these cutoffs against its own risk appetite --
    this is a clearly-stated, defensible starting policy, not a hidden rule.
    """
    if grade in ["AAA", "AA", "A", "BBB"]:
        return "Approve"
    elif grade in ["BB", "B"]:
        return "Approve with conditions (e.g. higher rate, lower limit)"
    else:
        return "Decline / refer for manual review"


def predict_individual(applicant: dict, version: str = "v1.0.0") -> dict:
    """
    Score a new individual borrower.

    Parameters
    ----------
    applicant : dict
        Raw feature values matching the individual model's expected inputs
        (see package['feature_names_raw'] for the full required list).
        Example keys: loan_amnt, int_rate, installment, annual_inc, dti,
        open_acc, total_acc, revol_util, revol_bal, emp_length_years,
        credit_history_years, loan_to_income, installment_to_income,
        total_derogatory_events, grade, home_ownership,
        verification_status, purpose, utilization_tier,
        recent_delinquency_flag, had_hardship, had_debt_settlement

    Returns
    -------
    dict with probability_of_default, risk_grade, recommendation, model_version
    """
    package = joblib.load(MODELS_DIR / f"individual_model_{version}.pkl")

    input_df = pd.DataFrame([applicant])[package["feature_names_raw"]]
    processed = package["preprocessor"].transform(input_df)

    pd_score = package["model"].predict_proba(processed)[0, 1]
    grade = _assign_grade(pd_score, package["risk_grade_edges"], package["risk_grade_labels"])

    return {
        "probability_of_default": round(float(pd_score), 4),
        "risk_grade": grade,
        "recommendation": _recommendation(grade),
        "model_version": version,
    }


def predict_sme(company: dict, version: str = "v1.0.0") -> dict:
    """
    Score a new SME / corporate borrower.

    Parameters
    ----------
    company : dict
        Raw feature values matching the SME model's expected inputs
        (see package['feature_names_raw'] for the full required list).
        Example keys: Current Ratio, Long-term Debt / Capital,
        Debt/Equity Ratio, Gross Margin, Operating Margin, EBIT Margin,
        EBITDA Margin, Pre-Tax Profit Margin, Net Profit Margin,
        Asset Turnover, ROE - Return On Equity, Return On Tangible Equity,
        ROA - Return On Assets, ROI - Return On Investment,
        Operating Cash Flow Per Share, Free Cash Flow Per Share,
        margin_spread, leverage_driven_returns_flag

    Returns
    -------
    dict with probability_of_default, risk_grade, recommendation, model_version

    Note: the underlying model predicts P(Investment Grade). We convert
    this to P(default-equivalent risk) = 1 - P(Investment Grade) so both
    prediction functions return a consistent "higher = riskier" score.
    """
    package = joblib.load(MODELS_DIR / f"sme_model_{version}.pkl")

    input_df = pd.DataFrame([company])[package["feature_names_raw"]]
    processed = package["preprocessor"].transform(input_df)

    p_investment_grade = package["model"].predict_proba(processed)[0, 1]
    risk_score = 1 - p_investment_grade

    grade = _assign_grade(risk_score, package["risk_grade_edges"], package["risk_grade_labels"])

    return {
        "probability_of_default": round(float(risk_score), 4),
        "risk_grade": grade,
        "recommendation": _recommendation(grade),
        "model_version": version,
    }


if __name__ == "__main__":
    # Quick manual test with a plausible example applicant/company
    example_individual = {
        "loan_amnt": 15000, "int_rate": 12.5, "installment": 450,
        "annual_inc": 65000, "dti": 18.0, "open_acc": 8, "total_acc": 20,
        "revol_util": 45.0, "revol_bal": 8000, "emp_length_years": 5,
        "credit_history_years": 12.0, "loan_to_income": 0.23,
        "installment_to_income": 0.083, "total_derogatory_events": 0,
        "grade": "B", "home_ownership": "MORTGAGE",
        "verification_status": "Verified", "purpose": "debt_consolidation",
        "utilization_tier": "moderate", "recent_delinquency_flag": 0,
        "had_hardship": 0, "had_debt_settlement": 0,
    }
    print("Individual example:", predict_individual(example_individual))

    example_company = {
        "Current Ratio": 1.8, "Long-term Debt / Capital": 0.35,
        "Debt/Equity Ratio": 0.6, "Gross Margin": 40.0, "Operating Margin": 15.0,
        "EBIT Margin": 14.0, "EBITDA Margin": 18.0, "Pre-Tax Profit Margin": 12.0,
        "Net Profit Margin": 8.0, "Asset Turnover": 1.1, "ROE - Return On Equity": 15.0,
        "Return On Tangible Equity": 16.0, "ROA - Return On Assets": 7.0,
        "ROI - Return On Investment": 10.0, "Operating Cash Flow Per Share": 3.5,
        "Free Cash Flow Per Share": 2.1, "margin_spread": 32.0,
        "leverage_driven_returns_flag": 0,
    }
    print("SME example:", predict_sme(example_company))
