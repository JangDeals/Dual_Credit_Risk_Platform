"""
streamlit_app/app.py

Dual Credit Risk Assessment Platform -- Streamlit application.

This file contains NO modeling logic. It only collects user input and
calls predict_individual() / predict_sme() from src/prediction_engine.py.
Keeping the UI and the model logic separate means either can change
independently without breaking the other.
"""

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable from streamlit_app/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.prediction_engine import predict_individual, predict_sme

st.set_page_config(page_title="Dual Credit Risk Platform", page_icon="\U0001F4CA", layout="centered")

# ---------------------------------------------------------------
# Landing / borrower type selection
# ---------------------------------------------------------------
st.title("Dual Credit Risk Assessment Platform")
st.caption("Predicting Probability of Default, Risk Grade, and Lending Recommendation.")

borrower_type = st.radio(
    "Select borrower type",
    ["Individual", "SME / Corporate"],
    horizontal=True,
)

st.divider()

# ---------------------------------------------------------------
# Individual borrower form
# ---------------------------------------------------------------
if borrower_type == "Individual":
    st.subheader("Individual Borrower Details")

    col1, col2 = st.columns(2)
    with col1:
        loan_amnt = st.number_input("Loan amount ($)", min_value=500, max_value=40000, value=15000, step=500)
        annual_inc = st.number_input("Annual income ($)", min_value=1000, max_value=1000000, value=65000, step=1000)
        int_rate = st.slider("Interest rate (%)", 5.0, 31.0, 12.5)
        installment = st.number_input("Monthly installment ($)", min_value=10.0, max_value=2000.0, value=450.0)
        dti = st.slider("Debt-to-income ratio (dti)", 0.0, 45.0, 18.0)
        revol_util = st.slider("Revolving credit utilization (%)", 0.0, 100.0, 45.0)
        revol_bal = st.number_input("Revolving balance ($)", min_value=0, max_value=200000, value=8000)
    with col2:
        emp_length_years = st.slider("Employment length (years)", 0, 10, 5)
        credit_history_years = st.slider("Credit history length (years)", 0.0, 60.0, 12.0)
        open_acc = st.number_input("Open credit accounts", min_value=0, max_value=50, value=8)
        total_acc = st.number_input("Total credit accounts", min_value=0, max_value=100, value=20)
        total_derogatory_events = st.number_input("Total derogatory events (delinquencies, public records, etc.)", min_value=0, max_value=30, value=0)
        grade = st.selectbox("LendingClub-style grade", ["A", "B", "C", "D", "E", "F", "G"])
        home_ownership = st.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        verification_status = st.selectbox("Income verification status", ["Verified", "Source Verified", "Not Verified"])
        purpose = st.selectbox("Loan purpose", ["debt_consolidation", "credit_card", "home_improvement", "medical", "small_business", "other"])
        utilization_tier = st.selectbox("Utilization tier", ["low", "moderate", "high", "maxed"])

    recent_delinquency_flag = st.checkbox("Delinquency in the last 2 years?")
    had_hardship = st.checkbox("Ever on a hardship plan?")
    had_debt_settlement = st.checkbox("Ever in debt settlement?")

    if st.button("Assess Individual Borrower", type="primary"):
        applicant = {
            "loan_amnt": loan_amnt, "int_rate": int_rate, "installment": installment,
            "annual_inc": annual_inc, "dti": dti, "open_acc": open_acc, "total_acc": total_acc,
            "revol_util": revol_util, "revol_bal": revol_bal, "emp_length_years": emp_length_years,
            "credit_history_years": credit_history_years,
            "loan_to_income": loan_amnt / annual_inc,
            "installment_to_income": installment / (annual_inc / 12),
            "total_derogatory_events": total_derogatory_events,
            "grade": grade, "home_ownership": home_ownership,
            "verification_status": verification_status, "purpose": purpose,
            "utilization_tier": utilization_tier,
            "recent_delinquency_flag": int(recent_delinquency_flag),
            "had_hardship": int(had_hardship), "had_debt_settlement": int(had_debt_settlement),
        }

        result = predict_individual(applicant)

        st.divider()
        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Probability of Default", f"{result['probability_of_default']*100:.1f}%")
        m2.metric("Risk Grade", result["risk_grade"])
        st.info(f"**Recommendation:** {result['recommendation']}")

# ---------------------------------------------------------------
# SME borrower form
# ---------------------------------------------------------------
else:
    st.subheader("SME / Corporate Borrower Details")
    st.caption("Enter the company's financial ratios (see the Data Dictionary in reports/ for definitions).")

    col1, col2 = st.columns(2)
    with col1:
        current_ratio = st.number_input("Current Ratio", value=1.8, step=0.1)
        ltd_capital = st.number_input("Long-term Debt / Capital", value=0.35, step=0.05)
        debt_equity = st.number_input("Debt/Equity Ratio", value=0.6, step=0.05)
        gross_margin = st.number_input("Gross Margin (%)", value=40.0, step=1.0)
        operating_margin = st.number_input("Operating Margin (%)", value=15.0, step=1.0)
        ebit_margin = st.number_input("EBIT Margin (%)", value=14.0, step=1.0)
        ebitda_margin = st.number_input("EBITDA Margin (%)", value=18.0, step=1.0)
        pretax_margin = st.number_input("Pre-Tax Profit Margin (%)", value=12.0, step=1.0)
    with col2:
        net_margin = st.number_input("Net Profit Margin (%)", value=8.0, step=1.0)
        asset_turnover = st.number_input("Asset Turnover", value=1.1, step=0.1)
        roe = st.number_input("ROE - Return On Equity (%)", value=15.0, step=1.0)
        rote = st.number_input("Return On Tangible Equity (%)", value=16.0, step=1.0)
        roa = st.number_input("ROA - Return On Assets (%)", value=7.0, step=1.0)
        roi = st.number_input("ROI - Return On Investment (%)", value=10.0, step=1.0)
        ocf_per_share = st.number_input("Operating Cash Flow Per Share ($)", value=3.5, step=0.1)
        fcf_per_share = st.number_input("Free Cash Flow Per Share ($)", value=2.1, step=0.1)

    if st.button("Assess SME Borrower", type="primary"):
        margin_spread = gross_margin - net_margin
        high_roe_flag = roe > 15  # simple runtime approximation of Module 11's percentile-based flag
        high_leverage_flag = debt_equity > 0.6
        leverage_driven_returns_flag = int(high_roe_flag and high_leverage_flag)

        company = {
            "Current Ratio": current_ratio, "Long-term Debt / Capital": ltd_capital,
            "Debt/Equity Ratio": debt_equity, "Gross Margin": gross_margin,
            "Operating Margin": operating_margin, "EBIT Margin": ebit_margin,
            "EBITDA Margin": ebitda_margin, "Pre-Tax Profit Margin": pretax_margin,
            "Net Profit Margin": net_margin, "Asset Turnover": asset_turnover,
            "ROE - Return On Equity": roe, "Return On Tangible Equity": rote,
            "ROA - Return On Assets": roa, "ROI - Return On Investment": roi,
            "Operating Cash Flow Per Share": ocf_per_share,
            "Free Cash Flow Per Share": fcf_per_share,
            "margin_spread": margin_spread,
            "leverage_driven_returns_flag": leverage_driven_returns_flag,
        }

        result = predict_sme(company)

        st.divider()
        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Risk Score (1 - P(Investment Grade))", f"{result['probability_of_default']*100:.1f}%")
        m2.metric("Risk Grade", result["risk_grade"])
        st.info(f"**Recommendation:** {result['recommendation']}")

st.divider()
st.caption(
    "Individual model: Logistic Regression, trained on a 10,000-row stratified sample "
    "of LendingClub data (test ROC-AUC 0.740). SME model: XGBoost, trained on the full "
    "corporate credit rating dataset (test ROC-AUC 0.955), validated against real agency "
    "ratings. See project documentation for full methodology and known limitations."
)
