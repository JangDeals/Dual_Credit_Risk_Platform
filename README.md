# Dual Credit Risk Assessment Platform

An end-to-end machine learning platform that predicts Probability of Default (PD), assigns a bank-style risk grade, and generates a lending recommendation for two distinct borrower types: **individual borrowers** and **SME / corporate borrowers**. Built as a portfolio project demonstrating credit risk modeling, explainable AI, and production-oriented ML engineering practices.

---

## 1. Business Problem

Lenders must answer the same underlying question for very different applicant types: *will this borrower repay?* Individual and corporate borrowers are assessed on fundamentally different signals — personal financial behavior versus company financial health — which is why real banks maintain separate retail and commercial credit models rather than a single unified scorecard.

This platform mirrors that structure: a single entry point routes each applicant to a specialized model built for their borrower type, while both models share a common output contract (PD, risk grade, recommendation) so they can be compared and deployed consistently.

## 2. Architecture

```
Select borrower type
         |
    -----------
    |         |
Individual   SME
credit       credit
model        model
    |         |
    -----------
         |
Streamlit app output:
PD, Risk Grade, SHAP explanation, Recommendation
```

## 3. Datasets

| Dataset | Source | Rows Used | Target |
|---|---|---|---|
| Individual (LendingClub) | [Kaggle](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) | 10,000-row stratified sample (development); full ~1.1GB file supported for production retraining | `target_default` (1 = Charged Off/Default, 0 = Fully Paid) |
| SME / Corporate | Corporate Credit Rating with Financial Ratios | 7,805 rows | `Binary Rating` (1 = Investment Grade, 0 = Speculative) |

## 4. Methodology

1. **Data Understanding & Quality Assessment** — built a full data dictionary (including manual documentation for 12 undocumented joint-application fields); found the SME dataset fully clean, the individual dataset with expected sparsity in rare-event fields (hardship plans, settlements).
2. **Cleaning** — dropped fully-empty columns, converted sparse hardship/settlement fields into binary flags rather than discarding them, fixed data types, and defined the target variable strictly from *resolved* loans only (Fully Paid/Charged Off/Default) to avoid label leakage from still-active loans.
3. **EDA** — validated that default rate rises monotonically across LendingClub's own A–G grades; identified `int_rate` as the strongest linear correlate of default.
4. **Feature Engineering** — created affordability ratios (`loan_to_income`, `installment_to_income`), a composite derogatory-events score, credit history length, and SME-specific features (`margin_spread`, a leverage-driven-returns flag). Not every engineered feature earned its place — see Limitations.
5. **Preprocessing** — explicitly removed all outcome-dependent columns (payment history, recoveries, `loan_status` itself) to prevent data leakage; built `ColumnTransformer` pipelines fit only on training data.
6. **Modeling** — trained Logistic Regression, Random Forest, and XGBoost for both borrower types; tuned tree models via randomized search.
7. **Evaluation** — compared models on ROC-AUC, precision/recall/F1, confusion matrices, and threshold sensitivity.
8. **Explainability** — SHAP global and local explanations for both final models.
9. **Risk Grading** — converted PD into a 10-tier AAA→D scale calibrated per-population (not borrowed from corporate bond scales, which assume much lower base default rates).

## 5. Results

| Model | Algorithm | Test ROC-AUC |
|---|---|---|
| Individual | Logistic Regression | 0.740 |
| SME | XGBoost (tuned) | 0.955 |

**Key finding:** for the individual dataset, tuned Random Forest (0.739) and XGBoost (0.738) performed statistically identically to plain Logistic Regression (0.740) — added model complexity bought no measurable accuracy. Logistic Regression was selected as the final individual model specifically because, at equal performance, its full coefficient-level interpretability is a genuine advantage for consumer lending, where explaining adverse decisions is often a legal requirement.

**Independent validation:** the SME model's risk grades were checked against real third-party agency ratings (S&P-style letter ratings) that were never used in training, and showed near-perfect monotonic agreement — from 100% investment-grade at AAA down to 0.6% at D.

## 6. Known Limitations

- Individual model developed on a 10,000-row sample of the full ~1.1GB LendingClub dataset; production deployment requires retraining on the full file (loading logic already supports this via `load_individual_data(sample=False)`).
- SME dataset lacks Quick Ratio and Interest Coverage Ratio — two ratios a full commercial underwriting process would include; estimating them from available columns was deliberately avoided since it would require fabricating unmeasured assumptions (e.g. an assumed interest rate) rather than using real data.
- The engineered `recent_delinquency_flag` and `leverage_driven_returns_flag` features showed negligible predictive separation in testing — kept in the pipeline for completeness and transparency, but did not meaningfully improve either model.
- Individual model shows two minor non-monotonic inversions in the middle risk grades (BB/BBB and CC/CCC), consistent with its moderate (0.74) discriminative power and modest test-set size per grade bucket.

## 7. How to Run

**Environment setup:**
```
conda create -n credit-risk-env python=3.11 -y
conda activate credit-risk-env
conda install -c conda-forge pandas numpy matplotlib scikit-learn xgboost shap joblib jupyter notebook ipykernel openpyxl -y
pip install streamlit
```

**Reproduce the analysis:**
Run the notebooks in `notebooks/` in order (`01_data_understanding.ipynb`, then `02_preprocessing_and_modeling.ipynb`).

**Launch the application:**
```
streamlit run streamlit_app/app.py
```

## 8. Project Structure

```
Dual_Credit_Risk_Platform/
├── data/
│   ├── raw/            # Original files (gitignored)
│   └── processed/      # Cleaned/engineered data (gitignored)
├── notebooks/           # Analysis notebooks, numbered by module
├── src/                 # Reusable code: data_loader.py, prediction_engine.py
├── models/               # Serialized models + registry (gitignored)
├── reports/              # Data dictionaries, figures
├── streamlit_app/        # app.py
└── README.md
```

## 9. Future Improvements

- Retrain the individual model on the full LendingClub dataset.
- Source a supplementary dataset with interest expense and revenue figures to compute Interest Coverage Ratio for the SME model.
- Investigate why delinquency-history features underperformed theoretical expectations — possibly requires a longer historical window than 2 years.
- Add model monitoring for production drift detection.
