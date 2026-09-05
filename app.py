# ------------------------------------------------------------------------------------
# Trustline Loan Default Risk Checker
# Streamlit frontend for the XGBoost model trained in the project notebook.
# Run locally with:  streamlit run app.py
# ------------------------------------------------------------------------------------
import json
import joblib
import pandas as pd
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Trustline Loan Default Risk", layout="wide")


# ---------- Load the saved artefacts ----------
# @st.cache_resource keeps the model in memory instead of reloading it on every interaction.
@st.cache_resource
def load_artefacts():
    model = joblib.load('trustline_model.joblib')
    with open('model_config.json') as f:
        config = json.load(f)
    return model, config


model, config = load_artefacts()
THRESHOLD = config['threshold']

# ---------- Header ----------
st.title("Trustline Loan Default Risk Checker")
st.write(
    "This tool estimates the probability that a loan applicant will default, using the "
    f"{config['model_name']} model trained on 255,347 historical Trustline loans."
)

# Show how the model performed, so the user knows what they are trusting.
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model ROC-AUC", f"{config['test_roc_auc']:.3f}")
col2.metric("Decision threshold", f"{THRESHOLD:.2f}")
col3.metric("Recall at threshold", f"{config['test_recall_at_threshold']:.1%}")
col4.metric("Precision at threshold", f"{config['test_precision_at_threshold']:.1%}")

st.divider()

tab1, tab2, tab3 = st.tabs(["Single applicant", "Score a CSV file", "About this model"])

# ==================================================================================
# TAB 1: score one applicant from a form
# ==================================================================================
with tab1:
    st.subheader("Applicant details")

    left, middle, right = st.columns(3)

    with left:
        st.markdown("**Personal**")
        age = st.slider("Age", 18, 70, 35)
        education = st.selectbox("Education", config['category_options']['Education'])
        marital_status = st.selectbox("Marital status", config['category_options']['MaritalStatus'])
        has_dependents = st.selectbox("Has dependents", config['category_options']['HasDependents'])

    with middle:
        st.markdown("**Financial**")
        income = st.number_input("Annual income", min_value=10000, max_value=200000,
                                 value=60000, step=1000)
        employment_type = st.selectbox("Employment type", config['category_options']['EmploymentType'])
        months_employed = st.slider("Months employed", 0, 120, 36)
        credit_score = st.slider("Credit score", 300, 850, 600)
        num_credit_lines = st.slider("Number of credit lines", 1, 4, 2)
        dti_ratio = st.slider("Debt-to-income ratio", 0.10, 0.90, 0.45, step=0.01)

    with right:
        st.markdown("**Loan**")
        loan_amount = st.number_input("Loan amount", min_value=5000, max_value=250000,
                                      value=100000, step=1000)
        interest_rate = st.slider("Interest rate (%)", 2.0, 25.0, 13.0, step=0.1)
        loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
        loan_purpose = st.selectbox("Loan purpose", config['category_options']['LoanPurpose'])
        has_mortgage = st.selectbox("Has mortgage", config['category_options']['HasMortgage'])
        has_cosigner = st.selectbox("Has co-signer", config['category_options']['HasCoSigner'])

    if st.button("Assess this application", type="primary"):
        # Build a one-row DataFrame using the exact column order the model expects.
        applicant = pd.DataFrame([{
            'Age': age, 'Income': income, 'LoanAmount': loan_amount,
            'CreditScore': credit_score, 'MonthsEmployed': months_employed,
            'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
            'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'Education': education,
            'EmploymentType': employment_type, 'MaritalStatus': marital_status,
            'HasMortgage': has_mortgage, 'HasDependents': has_dependents,
            'LoanPurpose': loan_purpose, 'HasCoSigner': has_cosigner
        }])[config['feature_order']]

        # The pipeline preprocesses the row itself, so raw values go straight in.
        probability = model.predict_proba(applicant)[0][1]

        st.divider()
        result_col, detail_col = st.columns([1, 2])

        with result_col:
            st.metric("Probability of default", f"{probability:.1%}")
            if probability >= THRESHOLD:
                st.error("REFER FOR REVIEW")
            else:
                st.success("APPROVE")

        with detail_col:
            st.write(f"Decision threshold: **{THRESHOLD:.2f}**")
            st.progress(min(float(probability), 1.0))
            st.caption(
                f"The average Trustline loan defaults {config['base_default_rate']:.1%} of the time. "
                f"This applicant scores {probability / config['base_default_rate']:.1f}x that rate."
            )

            # Flag the risk factors present, using the drivers found during the analysis.
            risk_factors = []
            if age < 35:
                risk_factors.append("Borrower is under 35")
            if credit_score < 500:
                risk_factors.append("Credit score below 500")
            if interest_rate > 18:
                risk_factors.append("Interest rate above 18%")
            if employment_type == 'Unemployed':
                risk_factors.append("Borrower is unemployed")
            if has_cosigner == 'No':
                risk_factors.append("No co-signer on the loan")
            if months_employed < 12:
                risk_factors.append("Less than a year in current employment")
            if loan_amount > income * 2:
                risk_factors.append("Loan is more than twice annual income")

            if risk_factors:
                st.write("**Risk factors present:**")
                for factor in risk_factors:
                    st.write("- " + factor)
            else:
                st.write("**No major risk factors flagged.**")

# ==================================================================================
# TAB 2: score a whole CSV file
# ==================================================================================
with tab2:
    st.subheader("Score a batch of applications")
    st.write("Upload a CSV with the same columns as the training data. LoanID is optional.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch):,} rows.")

        missing = [c for c in config['feature_order'] if c not in batch.columns]
        if missing:
            st.error(f"These required columns are missing: {missing}")
        else:
            scored = batch.copy()
            scored['Default_Probability'] = model.predict_proba(batch[config['feature_order']])[:, 1]
            scored['Decision'] = scored['Default_Probability'].apply(
                lambda p: 'REFER' if p >= THRESHOLD else 'APPROVE'
            )

            referred = (scored['Decision'] == 'REFER').sum()
            a, b = st.columns(2)
            a.metric("Applications referred", f"{referred:,}")
            b.metric("Share of the batch", f"{referred / len(scored):.1%}")

            st.dataframe(scored.sort_values('Default_Probability', ascending=False).head(50))

            st.download_button(
                "Download the scored file",
                scored.to_csv(index=False).encode('utf-8'),
                "scored_applications.csv",
                "text/csv"
            )

# ==================================================================================
# TAB 3: notes on the model
# ==================================================================================
with tab3:
    st.subheader("About this model")
    st.write(f"""
    **Model:** {config['model_name']}
    **Trained on:** 255,347 Trustline loan records, 80% used for training
    **Best parameters:** {config['best_params']}

    **How to read the output.** The model outputs a probability, not a verdict. An application is
    referred for review when that probability reaches {THRESHOLD:.2f}, a cut-off chosen by minimising
    the expected cost of the two mistakes, assuming a missed default costs about five times a wrongly
    rejected application.

    **What it catches.** At this threshold the model finds
    {config['test_recall_at_threshold']:.0%} of borrowers who go on to default, and
    {config['test_precision_at_threshold']:.0%} of the applications it refers do default.

    **What it cannot do.** The model only sees information available at application time. It has no
    payment history and no knowledge of what happens to a borrower after the loan is issued, so
    defaults triggered by job loss, illness or other later events are invisible to it. Use it to
    prioritise which applications a human reviews, not to make the final decision.
    """)
