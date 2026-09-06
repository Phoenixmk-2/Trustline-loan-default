"""
TrustLine Credit Decisioning
Loan default risk, priced and routed.

The model answers one question: how likely is this applicant to default?
This application answers the questions a lender actually acts on:
Score -> Price -> Decide -> Route -> Review -> Measure.
"""

import json
import os
import sqlite3
from datetime import datetime
from string import Template

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


PRODUCT_NAME = "TrustLine"
PRODUCT_TAGLINE = "Credit Decisioning & Portfolio Risk"
BUILT_BY = "Madu Miracle"

# --------------------------------------------------------------------------
# Credit risk assumptions
#
# These four numbers turn a probability into money. Every one of them is an
# assumption rather than a measurement, and each is surfaced in the interface
# so nobody mistakes it for a fact. A lender's finance team would replace them
# with its own figures; the arithmetic stays identical.
# --------------------------------------------------------------------------

# Loss given default: the share of principal not recovered when a loan goes
# bad. 45% is the Basel foundation-IRB figure for unsecured retail exposure.
LOSS_GIVEN_DEFAULT = 0.45

# Above this probability the application is declined outright rather than sent
# to an underwriter. Set so the decline queue stays small enough to be real.
DECLINE_THRESHOLD = 0.75

# From the notebook's threshold analysis: a missed default was costed at five
# times a wrongly rejected application. Shown on screen as the basis of the
# operating threshold.
COST_RATIO_MISSED_TO_REJECTED = 5.0

# Basis points added to the break-even rate when suggesting a price. This is
# the lender's required margin over expected loss, not a market rate.
TARGET_MARGIN_BPS = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(*candidates):
    # Return the first candidate that exists, otherwise the first candidate so
    # that any error message names a sensible expected location.
    for candidate in candidates:
        full_path = os.path.join(BASE_DIR, candidate)
        if os.path.exists(full_path):
            return full_path
    return os.path.join(BASE_DIR, candidates[0])


MODEL_FILE = find_file(
    "trustline_model.joblib",
    "models/trustline_model.joblib",
    "model/trustline_model.joblib",
)
CONFIG_FILE = find_file(
    "model_config.json",
    "models/model_config.json",
    "model/model_config.json",
)
DATABASE_FILE = os.path.join(BASE_DIR, "trustline_decisions.db")

# A sample of real applications for the queue and the portfolio views. Export
# one from the training CSV; the app scales to whatever size it finds.
SEED_FILE = find_file(
    "data/applications.csv",
    "data/Loan_default.csv",
    "Loan_default.csv",
    "applications.csv",
)

st.set_page_config(
    page_title=f"{PRODUCT_NAME} - Credit Decisioning",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# Theming
#
# .streamlit/config.toml defines a light and a dark theme. Streamlit exposes
# no CSS custom properties in the main app DOM, so the palette is resolved in
# Python and substituted into the stylesheet.
# ==========================================================================

# Decision colours use a fixed status palette in both themes, and every marker
# carries a shape and a word as well as a colour.
DECISION_COLORS = {"Decline": "#d03b3b", "Refer": "#fab219", "Approve": "#0ca30c"}
DECISION_ICONS = {"Decline": "▲", "Refer": "◆", "Approve": "●"}
DECISION_ORDER = ["Decline", "Refer", "Approve"]

DARK_PALETTE = {
    # Violet and fuchsia glows over a deep indigo gradient. The ground stays
    # dark and fairly low in chroma so the red/amber/green decision colours
    # and the blue/orange chart series keep their contrast against it.
    "appBackground": (
        "radial-gradient(circle at 15% 10%, rgba(139,92,246,.20), transparent 28%),"
        "radial-gradient(circle at 85% 15%, rgba(217,70,239,.15), transparent 28%),"
        "linear-gradient(135deg, #0c0718 0%, #16102e 48%, #1b1338 100%)"
    ),
    "sidebarBackground": "linear-gradient(180deg, #0e0920 0%, #191233 100%)",
    "text": "#f8fafc",
    "heading": "#ffffff",
    "muted": "#b8c3d6",
    "surface": "rgba(255,255,255,.07)",
    "surfaceSoft": "rgba(255,255,255,.06)",
    "border": "rgba(255,255,255,.10)",
    "shadow": "0 15px 45px rgba(0,0,0,.25)",
    "cardShadow": "0 10px 30px rgba(0,0,0,.18)",
    # Chart series stay blue and orange. On a violet ground this pair
    # separates far better than a green-family alternative (CVD delta-E 26.8
    # against 9.4), so the ground changed and the data colours did not.
    "series1": "#3987e5",
    "series2": "#d95926",
    "chartGrid": "#2b2350",
    "chartAxis": "#413769",
    "chartMuted": "#a99fce",
}

LIGHT_PALETTE = {
    "appBackground": (
        "radial-gradient(circle at 15% 10%, rgba(139,92,246,.11), transparent 28%),"
        "radial-gradient(circle at 85% 15%, rgba(217,70,239,.08), transparent 28%),"
        "linear-gradient(135deg, #faf8fe 0%, #f2eefb 48%, #ebe5f8 100%)"
    ),
    "sidebarBackground": "linear-gradient(180deg, #f2eefb 0%, #e6dff5 100%)",
    # Violet-tinted ink rather than slate, so the text belongs to the ground.
    "text": "#1a1330",
    "heading": "#1a1330",
    "muted": "#5b5280",
    "surface": "rgba(255,255,255,.80)",
    "surfaceSoft": "rgba(255,255,255,.68)",
    "border": "rgba(26,19,48,.12)",
    "shadow": "0 15px 45px rgba(26,19,48,.08)",
    "cardShadow": "0 10px 30px rgba(26,19,48,.07)",
    "series1": "#2a78d6",
    "series2": "#eb6834",
    "chartGrid": "#e7e0f5",
    "chartAxis": "#d4cae9",
    "chartMuted": "#6d6390",
}


def active_palette():
    # Reports the theme the viewer is actually using. Anything unexpected
    # falls back to dark, which is the configured default.
    try:
        theme_type = st.context.theme.type
    except Exception:
        theme_type = None
    return LIGHT_PALETTE if theme_type == "light" else DARK_PALETTE


PALETTE = active_palette()

STYLESHEET = Template(
    """
    <style>
    .stApp { background: $appBackground; color: $text; }
    .stApp, .stApp p, .stApp li, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] { color: $text; }
    [data-testid="stSidebar"] { background: $sidebarBackground; }
    [data-testid="stSidebar"] * { color: $text !important; }
    .block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1500px; }

    .masthead {
        display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
        padding: 0 0 6px 0; margin-bottom: 4px;
    }
    .masthead .brand {
        font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em;
        color: $heading; line-height: 1.1;
    }
    .masthead .tagline { font-size: .95rem; color: $muted; font-weight: 500; }
    .page-title {
        font-size: 1.35rem; font-weight: 700; color: $heading;
        margin: 4px 0 2px 0; letter-spacing: -.01em;
    }
    .page-sub { color: $muted; font-size: .9rem; margin-bottom: 18px; }
    .rule { height: 1px; background: $border; margin: 6px 0 20px 0; }

    .section-title {
        margin-top: 30px; margin-bottom: 2px; font-size: 1.05rem;
        font-weight: 700; color: $heading; letter-spacing: -.01em;
    }
    .section-question {
        margin-bottom: 12px; font-size: .84rem; color: $muted; font-style: italic;
    }

    .kpi {
        padding: 16px 18px; border-radius: 14px; background: $surface;
        border: 1px solid $border; box-shadow: $cardShadow; height: 100%;
    }
    .kpi-label {
        font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
        color: $muted; font-weight: 600; margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem; font-weight: 800; color: $heading;
        line-height: 1.1; letter-spacing: -.02em;
    }
    .kpi-note { font-size: .76rem; color: $muted; margin-top: 5px; }
    .kpi-critical { border-left: 4px solid #d03b3b; }
    .kpi-warning  { border-left: 4px solid #fab219; }
    .kpi-good     { border-left: 4px solid #0ca30c; }
    .kpi-neutral  { border-left: 4px solid $series1; }

    .decision-banner {
        padding: 20px 24px; border-radius: 16px; display: flex;
        align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 12px; margin: 6px 0 4px 0;
    }
    .decision-banner .db-left { display: flex; align-items: center; gap: 16px; }
    .decision-banner .db-icon { font-size: 1.6rem; }
    .decision-banner .db-verdict {
        font-size: 1.3rem; font-weight: 800; letter-spacing: .02em;
    }
    .decision-banner .db-ref { font-size: .85rem; color: $muted; }
    .decision-banner .db-pd { font-size: 2.3rem; font-weight: 800; line-height: 1; }

    .card {
        padding: 18px; border-radius: 14px; background: $surface;
        border: 1px solid $border; box-shadow: $cardShadow;
    }
    .muted { color: $muted; }
    .fineprint { font-size: .78rem; color: $muted; }

    .reason {
        padding: 12px 14px; border-radius: 10px; background: $surfaceSoft;
        border: 1px solid $border; border-left: 3px solid $series1;
        margin-bottom: 8px; font-size: .92rem;
    }
    .reason b { color: $heading; }

    div[data-testid="stMetric"] {
        background: $surfaceSoft; border: 1px solid $border;
        padding: 12px; border-radius: 12px;
    }
    </style>
    """
).safe_substitute(PALETTE)

st.markdown(STYLESHEET, unsafe_allow_html=True)


def style_chart(chart, height=260):
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0, fill=None)
        .configure_axis(
            labelColor=PALETTE["chartMuted"], titleColor=PALETTE["chartMuted"],
            gridColor=PALETTE["chartGrid"], domainColor=PALETTE["chartAxis"],
            tickColor=PALETTE["chartAxis"], labelFontSize=11,
            titleFontSize=11, titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=PALETTE["text"], titleColor=PALETTE["chartMuted"],
            labelFontSize=11, titleFontSize=11,
        )
    )


def section(title, question=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if question:
        st.markdown(f'<div class="section-question">{question}</div>',
                    unsafe_allow_html=True)


def kpi_tile(label, value, note="", tone="neutral"):
    st.markdown(
        f"""
        <div class="kpi kpi-{tone}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def money(amount):
    # Large sums read better abbreviated on a KPI tile.
    amount = float(amount)
    if abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:,.2f}M"
    if abs(amount) >= 10_000:
        return f"${amount/1_000:,.0f}K"
    return f"${amount:,.0f}"


def money_exact(amount):
    return f"${float(amount):,.0f}"


# ==========================================================================
# Database - the underwriting workflow
# ==========================================================================

CASE_STATUSES = ["Pending review", "In review", "Approved", "Declined", "Withdrawn"]

UNDERWRITERS = ["Unassigned", "A. Mensah", "B. Okonkwo", "C. Adeyemi", "D. Ncube"]

DECLINE_REASONS = [
    "Probability of default above policy limit",
    "Insufficient employment history",
    "Debt service capacity inadequate",
    "Loan amount excessive relative to income",
    "Credit score below policy minimum",
    "Applicant withdrew",
    "Other - see notes",
]


def get_connection():
    return sqlite3.connect(DATABASE_FILE, check_same_thread=False)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()
    # Every application the system has scored, with the decision it produced.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS applications (
            LoanID TEXT PRIMARY KEY,
            Age INTEGER, Income REAL, LoanAmount REAL, CreditScore INTEGER,
            MonthsEmployed INTEGER, NumCreditLines INTEGER, InterestRate REAL,
            LoanTerm INTEGER, DTIRatio REAL, Education TEXT, EmploymentType TEXT,
            MaritalStatus TEXT, HasMortgage TEXT, HasDependents TEXT,
            LoanPurpose TEXT, HasCoSigner TEXT,
            actual_default TEXT, source TEXT, received_at TEXT
        )
        """
    )
    # One row per scoring event, so a re-score leaves a history.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            score_id INTEGER PRIMARY KEY AUTOINCREMENT,
            LoanID TEXT,
            default_probability REAL,
            system_decision TEXT,
            threshold REAL,
            expected_loss REAL,
            expected_margin REAL,
            scored_at TEXT
        )
        """
    )
    # The underwriting queue. Snapshots the probability and the system's own
    # decision at the moment the case opened, so an override is measurable.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS underwriting (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            LoanID TEXT,
            opened_at TEXT,
            updated_at TEXT,
            default_probability REAL,
            system_decision TEXT,
            loan_amount REAL,
            expected_loss REAL,
            primary_reason TEXT,
            assigned_to TEXT,
            status TEXT,
            final_decision TEXT,
            decline_reason TEXT,
            overrode_system INTEGER DEFAULT 0,
            notes TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS system_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, level TEXT, event TEXT, LoanID TEXT, details TEXT
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scores_loan ON scores(LoanID)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uw_loan ON underwriting(LoanID)")
    connection.commit()
    connection.close()


def log_event(level, event, loan_id="", details="", connection=None):
    # A caller doing a bulk load passes its own connection and commits once.
    # Committing per row is what makes a cold start crawl on a cloud disk.
    owns = connection is None
    if owns:
        connection = get_connection()
    connection.execute(
        "INSERT INTO system_logs(timestamp, level, event, LoanID, details) "
        "VALUES (?, ?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), level, event, loan_id, details),
    )
    if owns:
        connection.commit()
        connection.close()


APPLICATION_COLUMNS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio", "Education",
    "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents",
    "LoanPurpose", "HasCoSigner",
]


def insert_application(row, loan_id, actual_default, source, connection=None):
    owns = connection is None
    if owns:
        connection = get_connection()
    values = [str(loan_id)]
    for column in APPLICATION_COLUMNS:
        value = row.get(column)
        if column in ("Age", "CreditScore", "MonthsEmployed",
                      "NumCreditLines", "LoanTerm"):
            values.append(int(float(value)))
        elif column in ("Income", "LoanAmount", "InterestRate", "DTIRatio"):
            values.append(float(value))
        else:
            values.append(str(value))
    values += [actual_default, source, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

    connection.execute(
        "INSERT OR REPLACE INTO applications (LoanID, " +
        ", ".join(APPLICATION_COLUMNS) +
        ", actual_default, source, received_at) VALUES (" +
        ", ".join(["?"] * (len(APPLICATION_COLUMNS) + 4)) + ")",
        values,
    )
    if owns:
        connection.commit()
        connection.close()


def save_score(loan_id, probability, decision, expected_loss,
               expected_margin, connection=None):
    owns = connection is None
    if owns:
        connection = get_connection()
    connection.execute(
        """
        INSERT INTO scores(LoanID, default_probability, system_decision,
                           threshold, expected_loss, expected_margin, scored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (loan_id, float(probability), decision, float(REFER_THRESHOLD),
         float(expected_loss), float(expected_margin),
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    if owns:
        connection.commit()
        connection.close()


def get_latest_case(loan_id):
    connection = get_connection()
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT * FROM underwriting WHERE LoanID = ? ORDER BY case_id DESC LIMIT 1",
        (loan_id,),
    ).fetchone()
    connection.close()
    return dict(row) if row else None


def open_case(loan_id, probability, system_decision, loan_amount, expected_loss,
              primary_reason, assigned_to="Unassigned", status="Pending review"):
    """Create a case, or hand back the live one so a double click cannot
    put the same application into the queue twice."""
    existing = get_latest_case(loan_id)
    if existing and existing["status"] not in ("Approved", "Declined", "Withdrawn"):
        return existing["case_id"], False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connection = get_connection()
    cursor = connection.execute(
        """
        INSERT INTO underwriting(
            LoanID, opened_at, updated_at, default_probability, system_decision,
            loan_amount, expected_loss, primary_reason, assigned_to, status,
            final_decision, decline_reason, overrode_system, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', 0, '')
        """,
        (loan_id, now, now, float(probability), system_decision,
         float(loan_amount), float(expected_loss), primary_reason,
         assigned_to, status),
    )
    case_id = cursor.lastrowid
    log_event("INFO", "Underwriting case opened", loan_id=loan_id,
              details=f"case={case_id}; system={system_decision}; "
                      f"pd={probability:.4f}", connection=connection)
    connection.commit()
    connection.close()
    return case_id, True


def update_case(case_id, **fields):
    if not fields:
        return
    fields["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [case_id]
    connection = get_connection()
    connection.execute(
        f"UPDATE underwriting SET {assignments} WHERE case_id = ?", values
    )
    log_event("INFO", "Underwriting case updated",
              details=f"case={case_id}; " +
                      "; ".join(f"{k}={v}" for k, v in fields.items()),
              connection=connection)
    connection.commit()
    connection.close()


# ==========================================================================
# Model layer
# ==========================================================================

FEATURE_LABELS = {
    "Age": "Age", "Income": "Annual income", "LoanAmount": "Loan amount",
    "CreditScore": "Credit score", "MonthsEmployed": "Months in employment",
    "NumCreditLines": "Credit lines open", "InterestRate": "Interest rate",
    "LoanTerm": "Loan term", "DTIRatio": "Debt-to-income ratio",
    "Education": "Education", "EmploymentType": "Employment type",
    "MaritalStatus": "Marital status", "HasMortgage": "Has mortgage",
    "HasDependents": "Has dependents", "LoanPurpose": "Loan purpose",
    "HasCoSigner": "Has co-signer",
}


@st.cache_resource
def load_artefacts():
    """Load the pipeline and its config.

    A pickled scikit-learn object is a frozen Python object graph. Loading it
    under a different scikit-learn version raises an AttributeError naming a
    private class, which tells the reader nothing. Catching it here turns that
    into a message that names the actual problem and the actual fix.
    """
    try:
        model = joblib.load(MODEL_FILE)
    except (AttributeError, ModuleNotFoundError, ImportError) as error:
        import sklearn
        raise RuntimeError(
            "The saved model could not be unpickled, which almost always means "
            "the installed library versions differ from the ones the model was "
            "trained with.\n\n"
            f"Installed scikit-learn: {sklearn.__version__}\n"
            f"Underlying error: {type(error).__name__}: {error}\n\n"
            "Fix: pin scikit-learn and xgboost in requirements.txt to the exact "
            "versions used for training, then reinstall. An unpinned floor such "
            "as 'scikit-learn>=1.3' lets a newer release install and break the "
            "pickle."
        ) from error

    with open(CONFIG_FILE) as handle:
        config = json.load(handle)
    return model, config


def classifier_of(model):
    # The saved artefact is a Pipeline; the classifier is its last step.
    try:
        return model.steps[-1][1]
    except Exception:
        return model


def preprocessor_of(model):
    try:
        return model.named_steps.get("preprocessor")
    except Exception:
        return None


def score_frame(frame, model, feature_order):
    """Score any number of applications in one call."""
    return model.predict_proba(frame[feature_order])[:, 1]


# ---------- The credit decision itself ----------

def decide(probability):
    """Three-way policy rather than approve/refer.

    Sending every flagged application to a human is only realistic while the
    flagged share stays small. Above DECLINE_THRESHOLD the model is confident
    enough that underwriter time is better spent elsewhere.
    """
    if probability >= DECLINE_THRESHOLD:
        return "Decline"
    if probability >= REFER_THRESHOLD:
        return "Refer"
    return "Approve"


def expected_loss(probability, loan_amount):
    """PD x LGD x EAD, the standard credit expected-loss identity.

    Probability of default times loss given default times exposure at default.
    This is the number a credit committee provisions against.
    """
    return float(probability) * LOSS_GIVEN_DEFAULT * float(loan_amount)


def expected_interest(loan_amount, interest_rate, loan_term_months):
    """Simple interest over the life of the loan.

    Deliberately simple rather than an amortising schedule: on a declining
    balance the lender earns less than this, so treating it as the ceiling
    keeps the margin figure conservative in the direction that matters.
    """
    return float(loan_amount) * (float(interest_rate) / 100.0) * (
        float(loan_term_months) / 12.0
    )


def break_even_rate(probability, loan_term_months):
    """The annual rate at which interest income just covers expected loss.

    Setting expected_interest equal to expected_loss and solving for the rate
    cancels the loan amount, so the answer depends only on the probability,
    the loss assumption and the term.
    """
    years = max(float(loan_term_months) / 12.0, 1e-9)
    return float(probability) * LOSS_GIVEN_DEFAULT / years * 100.0


def price_application(row, probability):
    """Everything the money side of the decision needs, in one dict."""
    amount = float(row.get("LoanAmount") or 0.0)
    rate = float(row.get("InterestRate") or 0.0)
    term = float(row.get("LoanTerm") or 12)

    loss = expected_loss(probability, amount)
    interest = expected_interest(amount, rate, term)
    breakeven = break_even_rate(probability, term)

    return {
        "loan_amount": amount,
        "interest_rate": rate,
        "loan_term": term,
        "expected_loss": loss,
        "expected_interest": interest,
        "expected_margin": interest - loss,
        "break_even_rate": breakeven,
        "suggested_rate": breakeven + TARGET_MARGIN_BPS / 100.0,
        "risk_adjusted_return": (interest - loss) / amount if amount else 0.0,
        "priced_correctly": rate >= breakeven,
    }


@st.cache_data(show_spinner=False)
def global_drivers(_model, _config):
    """Aggregate the classifier's importances over one-hot columns back to the
    original application fields, so the driver chart is the model's own
    ranking rather than a hand-written opinion."""
    classifier = classifier_of(_model)
    preprocessor = preprocessor_of(_model)
    try:
        importances = classifier.feature_importances_
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        return pd.DataFrame()

    fields = _config["feature_order"]
    totals = {}
    for name, importance in zip(names, importances):
        bare = name.split("__", 1)[-1]
        matched = None
        for field in fields:
            if bare == field or bare.startswith(field + "_"):
                if matched is None or len(field) > len(matched):
                    matched = field
        if matched:
            totals[matched] = totals.get(matched, 0.0) + float(importance)

    if not totals:
        return pd.DataFrame()
    total = sum(totals.values()) or 1.0
    frame = pd.DataFrame([
        {"Driver": FEATURE_LABELS.get(k, k), "Importance": v / total}
        for k, v in totals.items()
    ])
    return frame.sort_values("Importance", ascending=False).reset_index(drop=True)


# The adverse-action rule ladder.
#
# A lender that declines or conditions an application has to be able to say
# why, in specific terms. Running the full per-applicant sensitivity analysis
# across a whole queue is far too slow, so the queue uses this ladder, ordered
# at runtime by the model's own importances, and reports the highest-ranked
# adverse condition the applicant actually meets. The thresholds come from the
# notebook's exploratory findings.
ADVERSE_RULES = [
    ("Age", lambda r: float(r.get("Age") or 0) < 30,
     "Limited credit history for age"),
    ("InterestRate", lambda r: float(r.get("InterestRate") or 0) > 18,
     "Interest rate above 18%"),
    ("EmploymentType", lambda r: r.get("EmploymentType") == "Unemployed",
     "Applicant not currently employed"),
    ("MonthsEmployed", lambda r: float(r.get("MonthsEmployed") or 0) < 12,
     "Under one year in current employment"),
    ("Income", lambda r: float(r.get("LoanAmount") or 0) > float(r.get("Income") or 1) * 2,
     "Loan exceeds twice annual income"),
    ("CreditScore", lambda r: float(r.get("CreditScore") or 0) < 500,
     "Credit score below 500"),
    ("DTIRatio", lambda r: float(r.get("DTIRatio") or 0) > 0.6,
     "Debt-to-income ratio above 0.60"),
    ("HasCoSigner", lambda r: r.get("HasCoSigner") == "No",
     "No co-signer on the application"),
    ("LoanAmount", lambda r: float(r.get("LoanAmount") or 0) > 150000,
     "Loan amount above $150,000"),
]


def build_adverse_ladder(drivers_frame):
    if drivers_frame is None or drivers_frame.empty:
        return ADVERSE_RULES
    rank = {row["Driver"]: row["Importance"] for _, row in drivers_frame.iterrows()}
    return sorted(
        ADVERSE_RULES,
        key=lambda rule: rank.get(FEATURE_LABELS.get(rule[0], rule[0]), 0.0),
        reverse=True,
    )


def primary_reason(row, ladder):
    for _field, test, label in ladder:
        try:
            if test(row):
                return label
        except Exception:
            continue
    return "No single adverse factor"


def adverse_reasons(row, ladder, limit=4):
    reasons = []
    for _field, test, label in ladder:
        try:
            if test(row):
                reasons.append(label)
        except Exception:
            continue
        if len(reasons) >= limit:
            break
    return reasons


def applicant_sensitivity(row, model, config):
    """Per-applicant counterfactual: how far would the probability move if one
    field changed?

    This costs roughly 30 model calls, and Streamlit re-runs the whole script
    on every interaction, so without a cache the page burned that on every
    click. The wrapper turns the applicant into a hashable key and hands it to
    a cached worker; the same applicant is then scored once, not once per
    click. The underscore prefixes tell Streamlit not to try to hash the model
    and config, which it cannot do.
    """
    order = config["feature_order"]
    key = tuple((field, row[field]) for field in order)
    return _sensitivity_cached(key, model, config)


@st.cache_data(show_spinner=False, max_entries=256)
def _sensitivity_cached(row_key, _model, _config):
    row = dict(row_key)
    model, config = _model, _config
    order = config["feature_order"]
    base_frame = pd.DataFrame([{f: row[f] for f in order}])
    base = float(score_frame(base_frame, model, order)[0])
    signals = []

    ranges = config.get("numeric_ranges", {})
    options = config.get("category_options", {})

    for field in order:
        current = row[field]
        alternatives = []
        if field in config.get("numeric_cols", []):
            spec = ranges.get(field, {})
            low, high = spec.get("min"), spec.get("max")
            if low is None or high is None:
                continue
            value = float(current)
            span = (float(high) - float(low)) * 0.25
            alternatives = [
                max(float(low), value - span),
                min(float(high), value + span),
            ]
            # Whole-number fields should stay whole numbers.
            if float(value).is_integer() and float(low).is_integer():
                alternatives = [round(a) for a in alternatives]
        elif field in options:
            alternatives = [v for v in options[field] if v != current]

        best = None
        for alternative in alternatives:
            # Rebuild the row from a dict rather than assigning into the frame.
            # Newer pandas refuses to put a float into an int64 column instead
            # of silently upcasting, which raised on Age, CreditScore and the
            # other whole-number fields.
            candidate = base_frame.iloc[0].to_dict()
            candidate[field] = alternative
            modified = pd.DataFrame([candidate])
            try:
                moved = float(score_frame(modified, model, order)[0])
            except Exception:
                continue
            delta = base - moved
            if best is None or abs(delta) > abs(best[1]):
                best = (alternative, delta)

        if best is not None and abs(best[1]) > 0.001:
            alternative, delta = best
            display = (f"{float(alternative):,.2f}"
                       if field in config.get("numeric_cols", [])
                       else str(alternative))
            signals.append({
                "Factor": FEATURE_LABELS.get(field, field),
                "Current value": (f"{float(current):,.2f}"
                                  if field in config.get("numeric_cols", [])
                                  else str(current)),
                "Alternative": display,
                "Risk reduction": round(delta, 4),
            })

    if not signals:
        return pd.DataFrame()
    frame = pd.DataFrame(signals)
    frame["abs"] = frame["Risk reduction"].abs()
    return (frame.sort_values("abs", ascending=False)
            .drop(columns="abs").head(6).reset_index(drop=True))


# ==========================================================================
# Seeding and bootstrap
# ==========================================================================

def seed_applications(config):
    connection = get_connection()
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM applications").fetchone()[0]
        if count > 0 or not os.path.exists(SEED_FILE):
            return

        frame = pd.read_csv(SEED_FILE)
        missing = [c for c in APPLICATION_COLUMNS if c not in frame.columns]
        if missing:
            log_event("ERROR", "Seed file is missing required columns",
                      details=", ".join(missing), connection=connection)
            connection.commit()
            return

        if "LoanID" not in frame.columns:
            frame["LoanID"] = [f"L{i:06d}" for i in range(len(frame))]
        if "Default" not in frame.columns:
            frame["Default"] = "Unknown"

        for _, row in frame.iterrows():
            outcome = row["Default"]
            outcome = ("Yes" if str(outcome) in ("1", "Yes", "True")
                       else "No" if str(outcome) in ("0", "No", "False")
                       else "Unknown")
            insert_application(row, row["LoanID"], outcome,
                               "Historical application file",
                               connection=connection)

        log_event("INFO", "Application file loaded",
                  details=f"{len(frame)} applications from "
                          f"{os.path.basename(SEED_FILE)}.",
                  connection=connection)
        connection.commit()
    finally:
        connection.close()


def score_pending(model, config):
    connection = get_connection()
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT * FROM applications").fetchall()
        already = {r["LoanID"] for r in connection.execute(
            "SELECT DISTINCT LoanID FROM scores").fetchall()}
        pending = [dict(r) for r in rows if str(r["LoanID"]) not in already]
        if not pending:
            return

        order = config["feature_order"]
        frame = pd.DataFrame(pending)
        probabilities = score_frame(frame, model, order)

        for row, probability in zip(pending, probabilities):
            probability = float(probability)
            pricing = price_application(row, probability)
            save_score(str(row["LoanID"]), probability, decide(probability),
                       pricing["expected_loss"], pricing["expected_margin"],
                       connection=connection)

        log_event("INFO", "Applications scored",
                  details=f"{len(pending)} scored at threshold "
                          f"{REFER_THRESHOLD:.0%}.", connection=connection)
        connection.commit()
    finally:
        connection.close()


@st.cache_resource(show_spinner="Scoring the application book...")
def bootstrap():
    # Streamlit re-executes the whole script on every interaction, so this
    # runs once per server rather than once per click.
    initialize_database()
    try:
        model, config = load_artefacts()
    except Exception as error:
        log_event("ERROR", "Model artefacts could not be loaded",
                  details=str(error))
        return None, None, False, str(error)

    try:
        seed_applications(config)
        score_pending(model, config)
    except Exception as error:
        log_event("ERROR", "Seeding or scoring failed", details=str(error))
    return model, config, True, ""


# The operating threshold comes from the config the notebook wrote, so the app
# can never drift out of step with the analysis that chose it.
try:
    with open(CONFIG_FILE) as _handle:
        REFER_THRESHOLD = float(json.load(_handle).get("threshold", 0.60))
except Exception:
    REFER_THRESHOLD = 0.60

model, CONFIG, artefacts_loaded, artefact_error = bootstrap()

DRIVERS_FRAME = (global_drivers(model, CONFIG)
                 if artefacts_loaded else pd.DataFrame())
ADVERSE_LADDER = build_adverse_ladder(DRIVERS_FRAME)


# ==========================================================================
# Book data layer
# ==========================================================================

def data_version():
    """A short token that changes whenever the data changes, so cached tables
    refresh after a write without being stale for the life of the server."""
    connection = get_connection()
    try:
        scores = connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        cases = connection.execute(
            "SELECT COUNT(*) FROM underwriting").fetchone()[0]
        touched = connection.execute(
            "SELECT COALESCE(MAX(updated_at), '') FROM underwriting").fetchone()[0]
    finally:
        connection.close()
    return f"{scores}-{cases}-{touched}"


@st.cache_data(show_spinner=False)
def book(version):
    """One row per application: details, latest score, latest case."""
    connection = get_connection()
    try:
        frame = pd.read_sql_query(
            """
            SELECT a.*, s.default_probability, s.system_decision,
                   s.expected_loss, s.expected_margin, s.scored_at
            FROM applications a
            LEFT JOIN (
                SELECT s1.* FROM scores s1
                JOIN (SELECT LoanID, MAX(score_id) AS score_id
                      FROM scores GROUP BY LoanID) latest
                ON s1.score_id = latest.score_id
            ) s ON s.LoanID = a.LoanID
            """, connection)
        cases = pd.read_sql_query(
            """
            SELECT u1.LoanID, u1.status, u1.assigned_to, u1.final_decision,
                   u1.case_id, u1.overrode_system
            FROM underwriting u1
            JOIN (SELECT LoanID, MAX(case_id) AS case_id
                  FROM underwriting GROUP BY LoanID) latest
            ON u1.case_id = latest.case_id
            """, connection)
    finally:
        connection.close()

    if frame.empty:
        return frame

    frame = frame.merge(cases, on="LoanID", how="left")
    frame["default_probability"] = frame["default_probability"].fillna(0.0)
    frame["system_decision"] = frame["system_decision"].fillna("Approve")
    frame["status"] = frame["status"].fillna("No case")
    frame["assigned_to"] = frame["assigned_to"].fillna("Unassigned")
    frame["final_decision"] = frame["final_decision"].fillna("")

    for column in ("LoanAmount", "Income", "InterestRate", "DTIRatio"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in ("Age", "CreditScore", "MonthsEmployed", "LoanTerm"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    frame["expected_loss"] = (frame["default_probability"]
                              * LOSS_GIVEN_DEFAULT * frame["LoanAmount"])
    frame["expected_interest"] = (frame["LoanAmount"]
                                  * frame["InterestRate"] / 100.0
                                  * frame["LoanTerm"] / 12.0)
    frame["expected_margin"] = frame["expected_interest"] - frame["expected_loss"]
    frame["break_even_rate"] = (frame["default_probability"] * LOSS_GIVEN_DEFAULT
                                / (frame["LoanTerm"] / 12.0).replace(0, np.nan)
                                * 100.0).fillna(0.0)
    frame["underpriced"] = frame["InterestRate"] < frame["break_even_rate"]

    frame["primary_reason"] = [primary_reason(r, ADVERSE_LADDER)
                               for r in frame.to_dict(orient="records")]

    frame["credit_band"] = pd.cut(
        frame["CreditScore"], bins=[-1, 500, 600, 700, 800, 10000],
        labels=["<500", "500-600", "600-700", "700-800", "800+"]).astype(str)
    frame["amount_band"] = pd.cut(
        frame["LoanAmount"], bins=[-1, 25000, 75000, 125000, 175000, 10**9],
        labels=["<$25K", "$25-75K", "$75-125K", "$125-175K", "$175K+"]).astype(str)
    frame["age_band"] = pd.cut(
        frame["Age"], bins=[-1, 25, 35, 45, 55, 200],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"]).astype(str)
    frame["rate_band"] = pd.cut(
        frame["InterestRate"], bins=[-1, 8, 13, 18, 22, 100],
        labels=["<8%", "8-13%", "13-18%", "18-22%", "22%+"]).astype(str)
    return frame


@st.cache_data(show_spinner=False)
def case_log(version):
    connection = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM underwriting ORDER BY case_id DESC", connection)
    finally:
        connection.close()


def book_kpis(frame):
    total = len(frame)
    if total == 0:
        return {}

    approve = frame[frame["system_decision"] == "Approve"]
    refer = frame[frame["system_decision"] == "Refer"]
    declined = frame[frame["system_decision"] == "Decline"]

    known = frame[frame["actual_default"].isin(["Yes", "No"])]
    base_rate = (float((known["actual_default"] == "Yes").mean())
                 if len(known) else None)

    decided = frame[frame["final_decision"].isin(["Approved", "Declined"])]
    overrides = int(frame["overrode_system"].fillna(0).sum())

    return {
        "total": total,
        "known": len(known),
        "base_rate": base_rate,
        "approve": len(approve),
        "refer": len(refer),
        "decline": len(declined),
        "approval_rate": len(approve) / total,
        "referral_rate": len(refer) / total,
        "requested_ead": float(frame["LoanAmount"].sum()),
        "approved_ead": float(approve["LoanAmount"].sum()),
        "expected_loss_all": float(frame["expected_loss"].sum()),
        "expected_loss_approved": float(approve["expected_loss"].sum()),
        "expected_margin_approved": float(approve["expected_margin"].sum()),
        "weighted_pd_approved": (
            float((approve["default_probability"] * approve["LoanAmount"]).sum()
                  / approve["LoanAmount"].sum())
            if len(approve) and approve["LoanAmount"].sum() else 0.0),
        "underpriced": int(frame["underpriced"].sum()),
        "underpriced_ead": float(frame.loc[frame["underpriced"], "LoanAmount"].sum()),
        "open_cases": int((~frame["status"].isin(
            ["No case", "Approved", "Declined", "Withdrawn"])).sum()),
        "cases": int((frame["status"] != "No case").sum()),
        "decided": len(decided),
        "overrides": overrides,
        "override_rate": (overrides / len(decided)) if len(decided) else 0.0,
    }


# ==========================================================================
# Charts
#
# Single-series magnitude charts use one hue and no legend, because the title
# already names the series. Colour carries meaning only in the decision mix,
# where it uses the fixed status palette and every bar is directly labelled.
# ==========================================================================

CREDIT_ORDER = ["<500", "500-600", "600-700", "700-800", "800+"]
AMOUNT_ORDER = ["<$25K", "$25-75K", "$75-125K", "$125-175K", "$175K+"]
AGE_ORDER = ["18-25", "26-35", "36-45", "46-55", "56+"]
RATE_ORDER = ["<8%", "8-13%", "13-18%", "18-22%", "22%+"]


def _grouped(frame, dimension):
    grouped = (frame.groupby(dimension).agg(
        applications=("LoanID", "count"),
        flagged=("system_decision", lambda s: int((s != "Approve").sum())),
        ead=("LoanAmount", "sum"),
        loss=("expected_loss", "sum"),
        mean_pd=("default_probability", "mean"),
    ).reset_index())
    grouped["rate"] = (grouped["flagged"]
                       / grouped["applications"].replace(0, np.nan)).fillna(0.0)
    grouped = grouped.rename(columns={dimension: "category"})
    grouped["category"] = grouped["category"].astype(str)
    return grouped


def risk_rate_chart(frame, dimension, order=None, height=260):
    grouped = _grouped(frame, dimension)
    if grouped.empty:
        return None
    x = alt.X("category:N", title=None, sort=order if order else "-y",
              axis=alt.Axis(labelAngle=0, labelLimit=140))
    base = alt.Chart(grouped)
    bars = base.mark_bar(cornerRadiusEnd=4, size=38,
                         color=PALETTE["series1"]).encode(
        x=x,
        y=alt.Y("mean_pd:Q", title="Average default probability",
                axis=alt.Axis(format="%"), scale=alt.Scale(domainMin=0)),
        tooltip=[
            alt.Tooltip("category:N", title="Segment"),
            alt.Tooltip("mean_pd:Q", title="Average PD", format=".1%"),
            alt.Tooltip("applications:Q", title="Applications", format=","),
            alt.Tooltip("ead:Q", title="Amount requested", format="$,.0f"),
            alt.Tooltip("loss:Q", title="Expected loss", format="$,.0f"),
        ])
    labels = base.mark_text(dy=-8, fontSize=11, fontWeight=600,
                            color=PALETTE["text"]).encode(
        x=x, y=alt.Y("mean_pd:Q"), text=alt.Text("mean_pd:Q", format=".0%"))
    return style_chart(bars + labels, height=height)


def exposure_chart(frame, dimension, order=None, height=260):
    grouped = _grouped(frame, dimension)
    if grouped.empty:
        return None
    x = alt.X("category:N", title=None, sort=order if order else "-y",
              axis=alt.Axis(labelAngle=0, labelLimit=140))
    base = alt.Chart(grouped)
    bars = base.mark_bar(cornerRadiusEnd=4, size=38,
                         color=PALETTE["series2"]).encode(
        x=x,
        y=alt.Y("loss:Q", title="Expected loss ($)",
                scale=alt.Scale(domainMin=0)),
        tooltip=[
            alt.Tooltip("category:N", title="Segment"),
            alt.Tooltip("loss:Q", title="Expected loss", format="$,.0f"),
            alt.Tooltip("ead:Q", title="Amount requested", format="$,.0f"),
            alt.Tooltip("applications:Q", title="Applications", format=","),
        ])
    labels = base.mark_text(dy=-8, fontSize=11, fontWeight=600,
                            color=PALETTE["text"]).encode(
        x=x, y=alt.Y("loss:Q"), text=alt.Text("loss:Q", format="$,.0s"))
    return style_chart(bars + labels, height=height)


def decision_mix_chart(frame, height=210):
    counts = (frame.groupby("system_decision").agg(
        applications=("LoanID", "count"), ead=("LoanAmount", "sum"))
        .reindex(DECISION_ORDER).fillna(0).reset_index())
    counts["applications"] = counts["applications"].astype(int)
    total = counts["applications"].sum() or 1
    counts["share"] = counts["applications"] / total
    counts["label"] = counts.apply(
        lambda r: f"{r['applications']:,.0f}  ({r['share']:.0%})", axis=1)

    y = alt.Y("system_decision:N", title=None, sort=DECISION_ORDER,
              axis=alt.Axis(labelFontWeight=600, labelFontSize=12))
    base = alt.Chart(counts)
    bars = base.mark_bar(cornerRadiusEnd=4, size=30).encode(
        y=y,
        x=alt.X("applications:Q", title="Applications",
                scale=alt.Scale(domainMin=0)),
        color=alt.Color("system_decision:N",
                        scale=alt.Scale(domain=DECISION_ORDER,
                                        range=[DECISION_COLORS[d]
                                               for d in DECISION_ORDER]),
                        legend=None),
        tooltip=[
            alt.Tooltip("system_decision:N", title="Decision"),
            alt.Tooltip("applications:Q", title="Applications", format=","),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
            alt.Tooltip("ead:Q", title="Amount requested", format="$,.0f"),
        ])
    labels = base.mark_text(align="left", dx=6, fontSize=11, fontWeight=600,
                            color=PALETTE["text"]).encode(
        y=y, x=alt.X("applications:Q"), text=alt.Text("label:N"))
    return style_chart(bars + labels, height=height)


def drivers_chart(drivers_frame, height=260, top=8):
    if drivers_frame is None or drivers_frame.empty:
        return None
    frame = drivers_frame.head(top).copy()
    y = alt.Y("Driver:N", title=None, sort="-x",
              axis=alt.Axis(labelLimit=180))
    base = alt.Chart(frame)
    bars = base.mark_bar(cornerRadiusEnd=4, size=18,
                         color=PALETTE["series1"]).encode(
        y=y,
        x=alt.X("Importance:Q", title="Share of model importance",
                axis=alt.Axis(format="%"), scale=alt.Scale(domainMin=0)),
        tooltip=[alt.Tooltip("Driver:N"),
                 alt.Tooltip("Importance:Q", title="Model importance",
                             format=".1%")])
    labels = base.mark_text(align="left", dx=6, fontSize=11,
                            color=PALETTE["text"]).encode(
        y=y, x=alt.X("Importance:Q"),
        text=alt.Text("Importance:Q", format=".1%"))
    return style_chart(bars + labels, height=height)


def pricing_chart(frame, height=280):
    """Charged rate against the rate the risk requires. Points below the line
    are loans priced under their own expected loss."""
    sample = frame.sample(min(len(frame), 1500), random_state=1)
    points = alt.Chart(sample).mark_circle(size=34, opacity=0.55).encode(
        x=alt.X("break_even_rate:Q", title="Break-even rate required (%)",
                scale=alt.Scale(domainMin=0)),
        y=alt.Y("InterestRate:Q", title="Rate charged (%)",
                scale=alt.Scale(domainMin=0)),
        color=alt.Color("underpriced:N", title="Priced below risk",
                        scale=alt.Scale(domain=[False, True],
                                        range=[PALETTE["series1"], "#d03b3b"])),
        tooltip=[
            alt.Tooltip("LoanID:N", title="Application"),
            alt.Tooltip("default_probability:Q", title="PD", format=".1%"),
            alt.Tooltip("InterestRate:Q", title="Rate charged", format=".2f"),
            alt.Tooltip("break_even_rate:Q", title="Break-even", format=".2f"),
            alt.Tooltip("LoanAmount:Q", title="Amount", format="$,.0f"),
        ])
    limit = float(max(sample["break_even_rate"].max(),
                      sample["InterestRate"].max())) or 1.0
    line = alt.Chart(pd.DataFrame({"v": [0, limit]})).mark_line(
        color=PALETTE["chartMuted"], strokeDash=[5, 4],
        strokeWidth=1.5).encode(x="v:Q", y="v:Q")
    return style_chart(points + line, height=height)


def sample_note(frame):
    n = len(frame)
    if n >= 5000:
        return ""
    return (f"Computed over {n:,} applications in the current book. "
            "Segment cells are thin at this size - directional, not definitive.")


# ==========================================================================
# Navigation and roles
#
# This shapes what each role sees. It is NOT authentication: anyone can switch
# role from the sidebar. A real deployment would sit behind an identity
# provider, and credit decisioning would additionally need an audit trail
# tied to named users.
# ==========================================================================

ROLE_PAGES = {
    "Executive": ["Credit Risk Dashboard", "Portfolio Analytics"],
    "Credit Risk Manager": ["Credit Risk Dashboard", "Application Queue",
                            "Underwriting Cases", "Portfolio Analytics"],
    "Underwriter": ["Application Queue", "Applicant 360", "Underwriting Cases"],
    "Data Analyst": ["Credit Risk Dashboard", "Portfolio Analytics",
                     "Application Queue", "Applicant 360"],
    "Administrator": ["Credit Risk Dashboard", "Application Queue",
                      "Applicant 360", "Underwriting Cases",
                      "Portfolio Analytics", "System Logs"],
}

ROLE_BLURB = {
    "Executive": "Book quality and capital at risk.",
    "Credit Risk Manager": "Policy, queue health and override behaviour.",
    "Underwriter": "Your review queue and individual applications.",
    "Data Analyst": "Segment risk and model behaviour.",
    "Administrator": "Full access including system diagnostics.",
}

with st.sidebar:
    st.markdown(f"### 🏦 {PRODUCT_NAME}")
    st.caption(PRODUCT_TAGLINE)
    st.markdown("---")
    role = st.selectbox("Signed in as", list(ROLE_PAGES.keys()), index=0)
    st.caption(ROLE_BLURB[role])
    st.markdown("---")
    page = st.radio("Navigate", ROLE_PAGES[role], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Refer at {REFER_THRESHOLD:.0%} · Decline at "
               f"{DECLINE_THRESHOLD:.0%} · LGD {LOSS_GIVEN_DEFAULT:.0%}")
    st.caption(BUILT_BY)


st.markdown(
    f"""
    <div class="masthead">
        <span class="brand">{PRODUCT_NAME}</span>
        <span class="tagline">{PRODUCT_TAGLINE}</span>
    </div>
    <div class="rule"></div>
    """, unsafe_allow_html=True)

if not artefacts_loaded:
    st.error("The scoring model could not be loaded, so no decision can be made.")
    st.code(artefact_error)
    st.caption(f"Expected model at: {MODEL_FILE}")
    st.caption(f"Expected config at: {CONFIG_FILE}")
    st.stop()

VERSION = data_version()
BOOK = book(VERSION)

if BOOK.empty:
    st.warning(
        "No applications are loaded. The single-applicant scorer still works, "
        "but the queue and portfolio views need an application file.")
    st.caption(f"Expected an application CSV at: {SEED_FILE}")
    st.caption("It needs the 16 model columns; LoanID and Default are optional.")
    KPIS = {}
else:
    KPIS = book_kpis(BOOK)


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def require_book():
    if BOOK.empty:
        st.info("This view needs an application file. Load one to use it.")
        st.stop()


# ==========================================================================
# Credit Risk Dashboard
# ==========================================================================

if page == "Credit Risk Dashboard":
    require_book()
    page_header("Credit Risk Dashboard",
                "What we are lending, what it is likely to cost, and whether "
                "the price covers it.")

    row = st.columns(3)
    with row[0]:
        kpi_tile("Applications scored", f"{KPIS['total']:,}",
                 "Current application book", "neutral")
    with row[1]:
        kpi_tile("Auto-approval rate", f"{KPIS['approval_rate']:.1%}",
                 f"{KPIS['approve']:,} approved without review", "good")
    with row[2]:
        kpi_tile("Referred to underwriting", f"{KPIS['refer']:,}",
                 f"{KPIS['referral_rate']:.1%} of applications", "warning")

    row = st.columns(3)
    with row[0]:
        kpi_tile("Amount requested", money(KPIS["requested_ead"]),
                 "Exposure at default if all were written", "neutral")
    with row[1]:
        kpi_tile("Expected loss", money(KPIS["expected_loss_all"]),
                 f"PD × LGD × EAD at {LOSS_GIVEN_DEFAULT:.0%} loss given default",
                 "critical")
    with row[2]:
        kpi_tile("Loans priced below risk", f"{KPIS['underpriced']:,}",
                 f"{money(KPIS['underpriced_ead'])} of exposure", "critical")

    st.markdown(
        '<div class="fineprint" style="margin-top:10px;">'
        "Expected loss is probability of default times loss given default "
        f"times exposure at default. Loss given default is assumed at "
        f"{LOSS_GIVEN_DEFAULT:.0%}, the Basel foundation figure for unsecured "
        "retail, not a measured Trustline recovery rate. A loan is priced "
        "below risk when its interest rate is under the rate that would just "
        "cover its own expected loss."
        "</div>", unsafe_allow_html=True)

    if KPIS.get("base_rate") is not None and abs(KPIS["base_rate"] - 0.116) > 0.06:
        st.warning(
            f"**Book composition:** {KPIS['known']:,} of these applications have "
            f"a known outcome and {KPIS['base_rate']:.1%} of them defaulted, "
            "against 11.6% in the full training population. This sample is not "
            "representative, so the rates below describe this file rather than "
            "the business.")

    left, right = st.columns(2)
    with left:
        section("Decision mix", "How much of the book clears without a human?")
        chart = decision_mix_chart(BOOK)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
    with right:
        section("Default risk by credit band",
                "Is the score doing what a credit officer would expect?")
        chart = risk_rate_chart(BOOK, "credit_band", order=CREDIT_ORDER)
        if chart is not None:
            st.altair_chart(chart, width="stretch")

    left, right = st.columns(2)
    with left:
        section("What drives default", "Why is the model flagging people?")
        chart = drivers_chart(DRIVERS_FRAME)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption("The trained model's own feature importances, summed "
                       "over encoded columns back to application fields.")
        else:
            st.info("This model does not expose feature importances.")
    with right:
        section("Expected loss by loan size",
                "Where is the capital at risk concentrated?")
        chart = exposure_chart(BOOK, "amount_band", order=AMOUNT_ORDER)
        if chart is not None:
            st.altair_chart(chart, width="stretch")

    section("Largest exposures awaiting a decision",
            "Which applications should an underwriter open first?")
    queue = (BOOK[(BOOK["system_decision"] != "Approve")
                  & (BOOK["status"] == "No case")]
             .sort_values("expected_loss", ascending=False).head(10))
    if queue.empty:
        st.info("Nothing is waiting for review.")
    else:
        st.dataframe(pd.DataFrame({
            "Application": queue["LoanID"],
            "Decision": queue["system_decision"],
            "Default probability": queue["default_probability"],
            "Loan amount": queue["LoanAmount"],
            "Expected loss": queue["expected_loss"],
            "Primary reason": queue["primary_reason"],
        }), width="stretch", hide_index=True, column_config={
            "Default probability": st.column_config.ProgressColumn(
                "Default probability", format="%.0f%%", min_value=0, max_value=1),
            "Loan amount": st.column_config.NumberColumn(format="$%.0f"),
            "Expected loss": st.column_config.NumberColumn(format="$%.0f"),
        })
        st.caption("Ranked by expected loss rather than probability: a 70% risk "
                   "on $10,000 costs less than a 45% risk on $200,000.")

    note = sample_note(BOOK)
    if note:
        st.caption(note)


# ==========================================================================
# Application Queue
# ==========================================================================

elif page == "Application Queue":
    require_book()
    page_header("Application Queue",
                "The review worklist, ordered by what each decision is worth.")

    controls = st.columns([2, 2, 2, 3])
    with controls[0]:
        decisions = st.multiselect("System decision", DECISION_ORDER,
                                   default=["Decline", "Refer"])
    with controls[1]:
        statuses = st.multiselect("Case status",
                                  ["No case"] + CASE_STATUSES, default=[])
    with controls[2]:
        sort_by = st.selectbox("Sort by", ["Expected loss", "Loan amount",
                                           "Default probability", "Margin at risk"])
    with controls[3]:
        query = st.text_input("Find application", placeholder="e.g. L000123")

    view = BOOK.copy()
    if decisions:
        view = view[view["system_decision"].isin(decisions)]
    if statuses:
        view = view[view["status"].isin(statuses)]
    if query.strip():
        view = view[view["LoanID"].str.contains(query.strip(), case=False, na=False)]

    sort_column = {"Expected loss": "expected_loss", "Loan amount": "LoanAmount",
                   "Default probability": "default_probability",
                   "Margin at risk": "expected_margin"}[sort_by]
    view = view.sort_values(sort_column,
                            ascending=(sort_by == "Margin at risk"))

    summary = st.columns(4)
    with summary[0]:
        kpi_tile("In this view", f"{len(view):,}", "Matching the filters",
                 "neutral")
    with summary[1]:
        kpi_tile("Amount requested", money(view["LoanAmount"].sum()),
                 "Exposure in view", "neutral")
    with summary[2]:
        kpi_tile("Expected loss", money(view["expected_loss"].sum()),
                 "Risk-adjusted, in view", "warning")
    with summary[3]:
        kpi_tile("Unassigned", f"{int((view['status'] == 'No case').sum()):,}",
                 "No case opened yet", "critical")

    section("Worklist", "Which application should be opened first?")
    st.dataframe(pd.DataFrame({
        "Application": view["LoanID"],
        "Default probability": view["default_probability"],
        "Decision": view["system_decision"],
        "Loan amount": view["LoanAmount"],
        "Expected loss": view["expected_loss"],
        "Rate charged": view["InterestRate"],
        "Break-even rate": view["break_even_rate"],
        "Primary reason": view["primary_reason"],
        "Status": view["status"],
        "Owner": view["assigned_to"],
    }), width="stretch", hide_index=True, height=460, column_config={
        "Default probability": st.column_config.ProgressColumn(
            "Default probability", format="%.0f%%", min_value=0, max_value=1),
        "Loan amount": st.column_config.NumberColumn(format="$%.0f"),
        "Expected loss": st.column_config.NumberColumn(format="$%.0f"),
        "Rate charged": st.column_config.NumberColumn(format="%.2f%%"),
        "Break-even rate": st.column_config.NumberColumn(format="%.2f%%"),
    })
    st.caption("Break-even rate is the rate at which interest income would just "
               "cover this loan's own expected loss. Where the rate charged is "
               "lower, the loan loses money in expectation even if it performs.")

    if not view.empty:
        section("Open a case", "Move an application into underwriting.")
        picker = st.columns([3, 2, 2, 2])
        with picker[0]:
            chosen = st.selectbox("Application", view["LoanID"].tolist(),
                                  key="queue_pick")
        with picker[1]:
            owner = st.selectbox("Assign to", UNDERWRITERS, index=1)
        with picker[2]:
            status = st.selectbox("Opening status", CASE_STATUSES[:2])
        with picker[3]:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Open case", type="primary", width="stretch"):
                record = view[view["LoanID"] == chosen].iloc[0]
                _case, created = open_case(
                    chosen, record["default_probability"],
                    record["system_decision"], record["LoanAmount"],
                    record["expected_loss"], record["primary_reason"],
                    assigned_to=owner, status=status)
                if created:
                    st.success(f"Case opened for {chosen}, assigned to {owner}.")
                else:
                    st.info(f"{chosen} already has an open case.")
                st.rerun()


# ==========================================================================
# Applicant 360
# ==========================================================================

elif page == "Applicant 360":
    page_header("Applicant 360",
                "Score an application, price it, and record the decision.")

    def render_decision(loan_id, probability, row, key_prefix):
        verdict = decide(probability)
        colour = DECISION_COLORS[verdict]
        icon = DECISION_ICONS[verdict]
        pricing = price_application(row, probability)

        st.markdown(
            f"""
            <div class="decision-banner" style="
                background:{colour}1f; border:1px solid {colour}66;
                border-left:6px solid {colour};">
                <div class="db-left">
                    <div class="db-icon" style="color:{colour};">{icon}</div>
                    <div>
                        <div class="db-verdict" style="color:{colour};">
                            {verdict.upper()}
                        </div>
                        <div class="db-ref">Application {loan_id}</div>
                    </div>
                </div>
                <div class="db-pd" style="color:{colour};">{probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        base = CONFIG.get("base_default_rate", 0.116)
        st.caption(
            f"Probability of default {probability:.1%}, against a book average "
            f"of {base:.1%}. That is {probability / base:.1f} times the average "
            f"applicant. Refer at {REFER_THRESHOLD:.0%}, decline at "
            f"{DECLINE_THRESHOLD:.0%}.")

        figures = st.columns(4)
        with figures[0]:
            kpi_tile("Loan amount", money_exact(pricing["loan_amount"]),
                     "Exposure at default", "neutral")
        with figures[1]:
            kpi_tile("Expected loss", money_exact(pricing["expected_loss"]),
                     f"PD × {LOSS_GIVEN_DEFAULT:.0%} LGD × amount", "critical")
        with figures[2]:
            kpi_tile("Expected interest", money_exact(pricing["expected_interest"]),
                     f"{pricing['interest_rate']:.2f}% over "
                     f"{int(pricing['loan_term'])} months", "neutral")
        with figures[3]:
            tone = "good" if pricing["expected_margin"] > 0 else "critical"
            kpi_tile("Expected margin", money_exact(pricing["expected_margin"]),
                     "Interest less expected loss", tone)

        section("Is this loan priced for its risk?",
                "Would it make money in expectation?")
        if pricing["priced_correctly"]:
            st.success(
                f"Priced above risk. This application needs at least "
                f"{pricing['break_even_rate']:.2f}% to cover its own expected "
                f"loss and is being charged {pricing['interest_rate']:.2f}%. "
                f"Risk-adjusted return {pricing['risk_adjusted_return']:.2%} "
                "of principal.")
        else:
            st.error(
                f"Priced below risk. This application needs "
                f"{pricing['break_even_rate']:.2f}% to break even and is only "
                f"being charged {pricing['interest_rate']:.2f}%. At a "
                f"{TARGET_MARGIN_BPS} basis point target margin the price "
                f"would need to be {pricing['suggested_rate']:.2f}%.")

        section("Adverse factors", "What would we tell the applicant?")
        reasons = adverse_reasons(row, ADVERSE_LADDER)
        if not reasons:
            st.info("No adverse factors met the policy thresholds.")
        else:
            for reason in reasons:
                st.markdown(f"<div class='reason'><b>{reason}</b></div>",
                            unsafe_allow_html=True)
            st.caption(
                "Ordered by the model's own feature importance. A lender "
                "declining an application generally has to give specific "
                "reasons, and these are the candidate reason codes.")

        section("What would move this decision",
                "Which single change matters most?")
        signals = applicant_sensitivity(row, model, CONFIG)
        if signals.empty:
            st.info("No single field moves this application's score materially.")
        else:
            st.dataframe(signals, width="stretch", hide_index=True,
                         column_config={"Risk reduction":
                             st.column_config.NumberColumn(
                                 "Change in default probability",
                                 format="%+.1f%%")})
            st.caption("Each row re-scores the application with one field "
                       "changed. A positive value means the change lowers the "
                       "probability of default. This describes the model, not "
                       "a causal claim about the borrower.")

        section("Underwriting", "Record the decision.")
        existing = get_latest_case(loan_id)
        if existing:
            st.markdown(
                f"<div class='card'><b>Case #{existing['case_id']}</b> · "
                f"status <b>{existing['status']}</b> · owner "
                f"<b>{existing['assigned_to']}</b><br>"
                f"<span class='muted'>System said {existing['system_decision']}"
                f"{' · final: ' + existing['final_decision'] if existing['final_decision'] else ''}"
                f"</span></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        buttons = st.columns(4)
        with buttons[0]:
            if st.button("Send to underwriting", type="primary", width="stretch",
                         key=f"{key_prefix}_send"):
                open_case(loan_id, probability, verdict, pricing["loan_amount"],
                          pricing["expected_loss"],
                          primary_reason(row, ADVERSE_LADDER),
                          assigned_to=UNDERWRITERS[1])
                st.rerun()
        with buttons[1]:
            if st.button("Approve", width="stretch", key=f"{key_prefix}_approve"):
                case = existing or {"case_id": open_case(
                    loan_id, probability, verdict, pricing["loan_amount"],
                    pricing["expected_loss"],
                    primary_reason(row, ADVERSE_LADDER))[0]}
                update_case(case["case_id"], status="Approved",
                            final_decision="Approved",
                            overrode_system=int(verdict != "Approve"))
                st.rerun()
        with buttons[2]:
            if st.button("Decline", width="stretch", key=f"{key_prefix}_decline"):
                case = existing or {"case_id": open_case(
                    loan_id, probability, verdict, pricing["loan_amount"],
                    pricing["expected_loss"],
                    primary_reason(row, ADVERSE_LADDER))[0]}
                update_case(case["case_id"], status="Declined",
                            final_decision="Declined",
                            decline_reason=(reasons[0] if reasons
                                            else DECLINE_REASONS[0]),
                            overrode_system=int(verdict == "Approve"))
                st.rerun()
        with buttons[3]:
            if st.button("Request more information", width="stretch",
                         key=f"{key_prefix}_info"):
                case = existing or {"case_id": open_case(
                    loan_id, probability, verdict, pricing["loan_amount"],
                    pricing["expected_loss"],
                    primary_reason(row, ADVERSE_LADDER))[0]}
                update_case(case["case_id"], status="In review")
                st.rerun()

        st.caption("Decisions are stored in the application database. On a "
                   "hosted free tier that database resets when the container "
                   "restarts.")

    lookup, scorer = st.tabs(["Look up an application", "Score a new application"])

    with lookup:
        if BOOK.empty:
            st.info("No application file is loaded. Use the other tab to score "
                    "an application by hand.")
        else:
            options = BOOK.sort_values("expected_loss",
                                       ascending=False)["LoanID"].tolist()
            chosen = st.selectbox("Application", options,
                                  help="Ordered by expected loss.")
            record = BOOK[BOOK["LoanID"] == chosen].iloc[0].to_dict()
            render_decision(chosen, float(record["default_probability"]),
                            record, "lookup")

    with scorer:
        options = CONFIG.get("category_options", {})
        ranges = CONFIG.get("numeric_ranges", {})

        def rng(field, fallback_min, fallback_max, fallback_default):
            spec = ranges.get(field, {})
            return (float(spec.get("min", fallback_min)),
                    float(spec.get("max", fallback_max)),
                    float(spec.get("default", fallback_default)))

        with st.form("score_form"):
            top = st.columns([3, 2])
            with top[0]:
                new_id = st.text_input("Application reference",
                                       placeholder="e.g. L900001")
            with top[1]:
                st.markdown("<div style='height:28px'></div>",
                            unsafe_allow_html=True)
                submitted = st.form_submit_button("Score application",
                                                  type="primary",
                                                  width="stretch")

            st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Applicant**")
                lo, hi, dv = rng("Age", 18, 70, 35)
                age = st.slider("Age", int(lo), int(hi), int(dv))
                education = st.selectbox("Education",
                                         options.get("Education", ["Bachelor's"]))
                marital = st.selectbox("Marital status",
                                       options.get("MaritalStatus", ["Single"]))
                dependents = st.selectbox("Has dependents",
                                          options.get("HasDependents", ["No", "Yes"]))
                mortgage = st.selectbox("Has mortgage",
                                        options.get("HasMortgage", ["No", "Yes"]))
            with c2:
                st.markdown("**Finances**")
                lo, hi, dv = rng("Income", 15000, 200000, 60000)
                income = st.number_input("Annual income", float(lo), float(hi),
                                         float(dv), step=1000.0)
                employment = st.selectbox("Employment type",
                                          options.get("EmploymentType",
                                                      ["Full-time"]))
                lo, hi, dv = rng("MonthsEmployed", 0, 120, 36)
                months = st.slider("Months in employment", int(lo), int(hi), int(dv))
                lo, hi, dv = rng("CreditScore", 300, 850, 600)
                credit = st.slider("Credit score", int(lo), int(hi), int(dv))
                lo, hi, dv = rng("NumCreditLines", 1, 4, 2)
                lines = st.slider("Credit lines open", int(lo), int(hi), int(dv))
                lo, hi, dv = rng("DTIRatio", 0.1, 0.9, 0.45)
                dti = st.slider("Debt-to-income ratio", float(lo), float(hi),
                                float(dv), step=0.01)
            with c3:
                st.markdown("**The loan**")
                lo, hi, dv = rng("LoanAmount", 5000, 250000, 100000)
                amount = st.number_input("Loan amount", float(lo), float(hi),
                                         float(dv), step=1000.0)
                lo, hi, dv = rng("InterestRate", 2.0, 25.0, 13.0)
                rate = st.slider("Interest rate (%)", float(lo), float(hi),
                                 float(dv), step=0.1)
                term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60],
                                    index=2)
                purpose = st.selectbox("Loan purpose",
                                       options.get("LoanPurpose", ["Other"]))
                cosigner = st.selectbox("Has co-signer",
                                        options.get("HasCoSigner", ["No", "Yes"]))

        if submitted:
            if not new_id.strip():
                st.warning("Enter an application reference before scoring.")
            else:
                row = {"Age": age, "Income": income, "LoanAmount": amount,
                       "CreditScore": credit, "MonthsEmployed": months,
                       "NumCreditLines": lines, "InterestRate": rate,
                       "LoanTerm": term, "DTIRatio": dti, "Education": education,
                       "EmploymentType": employment, "MaritalStatus": marital,
                       "HasMortgage": mortgage, "HasDependents": dependents,
                       "LoanPurpose": purpose, "HasCoSigner": cosigner}
                order = CONFIG["feature_order"]
                probability = float(score_frame(pd.DataFrame([row]), model, order)[0])
                pricing = price_application(row, probability)
                insert_application(row, new_id.strip(), "Unknown",
                                   "Scored in application")
                save_score(new_id.strip(), probability, decide(probability),
                           pricing["expected_loss"], pricing["expected_margin"])
                log_event("INFO", "Application scored", loan_id=new_id.strip(),
                          details=f"pd={probability:.4f}; "
                                  f"decision={decide(probability)}")
                row["LoanID"] = new_id.strip()
                st.session_state["scored"] = {"id": new_id.strip(),
                                              "probability": probability,
                                              "row": row}

        if "scored" in st.session_state:
            saved = st.session_state["scored"]
            render_decision(saved["id"], saved["probability"], saved["row"],
                            "scored")


# ==========================================================================
# Underwriting Cases
# ==========================================================================

elif page == "Underwriting Cases":
    require_book()
    page_header("Underwriting Cases",
                "Model decision → human review → final outcome. Where the "
                "policy is tested.")

    cases = case_log(VERSION)

    row = st.columns(4)
    with row[0]:
        kpi_tile("Cases opened", f"{len(cases):,}", "All time", "neutral")
    with row[1]:
        kpi_tile("In the queue", f"{KPIS['open_cases']:,}", "Not yet decided",
                 "warning")
    with row[2]:
        kpi_tile("Decided", f"{KPIS['decided']:,}", "Approved or declined",
                 "neutral")
    with row[3]:
        tone = "critical" if KPIS["override_rate"] > 0.30 else "good"
        kpi_tile("Override rate", f"{KPIS['override_rate']:.0%}",
                 f"{KPIS['overrides']:,} decisions against the model", tone)

    st.markdown(
        '<div class="fineprint" style="margin-top:10px;">'
        "An override is an underwriter approving what the model would decline "
        "or referred, or declining what it would approve. A rate near zero "
        "suggests the review adds nothing; a very high rate suggests the "
        "policy thresholds are wrong. The number is a check on both."
        "</div>", unsafe_allow_html=True)

    if cases.empty:
        st.info("No cases yet. Open one from the Application Queue or from a "
                "scored application on Applicant 360.")
    else:
        section("Case pipeline", "Where does the work sit?")
        pipeline = (cases.groupby("status")
                    .agg(cases=("case_id", "count"),
                         exposure=("loan_amount", "sum"))
                    .reindex(CASE_STATUSES).fillna(0).reset_index())
        pipeline["cases"] = pipeline["cases"].astype(int)
        y = alt.Y("status:N", title=None, sort=CASE_STATUSES)
        base = alt.Chart(pipeline)
        bars = base.mark_bar(cornerRadiusEnd=4, size=26,
                             color=PALETTE["series1"]).encode(
            y=y, x=alt.X("cases:Q", title="Cases",
                         scale=alt.Scale(domainMin=0)),
            tooltip=[alt.Tooltip("status:N", title="Status"),
                     alt.Tooltip("cases:Q", title="Cases"),
                     alt.Tooltip("exposure:Q", title="Exposure",
                                 format="$,.0f")])
        labels = base.mark_text(align="left", dx=6, fontSize=11, fontWeight=600,
                                color=PALETTE["text"]).encode(
            y=y, x=alt.X("cases:Q"), text=alt.Text("cases:Q"))
        st.altair_chart(style_chart(bars + labels, height=220), width="stretch")

        section("Case book", "Every review and how it ended.")
        display = cases.copy()
        display["Override"] = display["overrode_system"].map({1: "Yes", 0: "-"})
        st.dataframe(pd.DataFrame({
            "Case": display["case_id"],
            "Application": display["LoanID"],
            "PD at open": display["default_probability"],
            "System said": display["system_decision"],
            "Loan amount": display["loan_amount"],
            "Expected loss": display["expected_loss"],
            "Primary reason": display["primary_reason"],
            "Owner": display["assigned_to"],
            "Status": display["status"],
            "Final": display["final_decision"],
            "Override": display["Override"],
            "Opened": display["opened_at"],
        }), width="stretch", hide_index=True, column_config={
            "PD at open": st.column_config.ProgressColumn(
                "PD at open", format="%.0f%%", min_value=0, max_value=1),
            "Loan amount": st.column_config.NumberColumn(format="$%.0f"),
            "Expected loss": st.column_config.NumberColumn(format="$%.0f"),
        })
        st.caption("The probability and system decision are recorded as they "
                   "stood when the case opened, not looked up later, so an "
                   "override stays measurable after the model is retrained.")

        section("Update a case", "Record the underwriting outcome.")
        edit = st.columns([2, 2, 2, 3])
        with edit[0]:
            case_id = st.selectbox("Case", cases["case_id"].tolist())
        current = cases[cases["case_id"] == case_id].iloc[0]
        with edit[1]:
            new_status = st.selectbox(
                "Status", CASE_STATUSES,
                index=CASE_STATUSES.index(current["status"])
                if current["status"] in CASE_STATUSES else 0)
        with edit[2]:
            new_owner = st.selectbox(
                "Owner", UNDERWRITERS,
                index=UNDERWRITERS.index(current["assigned_to"])
                if current["assigned_to"] in UNDERWRITERS else 0)
        with edit[3]:
            reason = st.selectbox("Reason if declined", DECLINE_REASONS)

        if st.button("Save case update", type="primary"):
            final = ("Approved" if new_status == "Approved"
                     else "Declined" if new_status == "Declined" else "")
            override = 0
            if final == "Approved" and current["system_decision"] != "Approve":
                override = 1
            if final == "Declined" and current["system_decision"] == "Approve":
                override = 1
            update_case(int(case_id), status=new_status, assigned_to=new_owner,
                        final_decision=final,
                        decline_reason=reason if final == "Declined" else "",
                        overrode_system=override)
            st.success(f"Case #{case_id} updated.")
            st.rerun()


# ==========================================================================
# Portfolio Analytics
# ==========================================================================

elif page == "Portfolio Analytics":
    require_book()
    page_header("Portfolio Analytics",
                "Who defaults, what it costs, whether we price it, and how "
                "well the model holds up.")

    tabs = st.tabs(["Who defaults", "What it costs", "Are we pricing it",
                    "Does the model hold up"])

    with tabs[0]:
        left, right = st.columns(2)
        with left:
            section("Default risk by credit band",
                    "Does the score track the bureau score?")
            chart = risk_rate_chart(BOOK, "credit_band", order=CREDIT_ORDER)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
        with right:
            section("Default risk by employment type",
                    "Does income security predict repayment?")
            chart = risk_rate_chart(BOOK, "EmploymentType")
            if chart is not None:
                st.altair_chart(chart, width="stretch")

        left, right = st.columns(2)
        with left:
            section("Default risk by age band",
                    "Where in the life cycle does risk sit?")
            chart = risk_rate_chart(BOOK, "age_band", order=AGE_ORDER)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
        with right:
            section("Default risk by loan purpose",
                    "Is any product line underperforming?")
            chart = risk_rate_chart(BOOK, "LoanPurpose")
            if chart is not None:
                st.altair_chart(chart, width="stretch")

        section("Default risk by rate charged",
                "Are the rates we set already tracking the risk?")
        chart = risk_rate_chart(BOOK, "rate_band", order=RATE_ORDER)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption("A rising line here is partly circular: the lender "
                       "already charges risky applicants more, so rate is both "
                       "a cause and a symptom of risk.")

    with tabs[1]:
        row = st.columns(4)
        with row[0]:
            kpi_tile("Amount requested", money(KPIS["requested_ead"]),
                     "All applications", "neutral")
        with row[1]:
            kpi_tile("Expected loss, whole book",
                     money(KPIS["expected_loss_all"]),
                     "If every application were written", "critical")
        with row[2]:
            kpi_tile("Expected loss, auto-approved",
                     money(KPIS["expected_loss_approved"]),
                     "What clears without review", "warning")
        with row[3]:
            tone = "good" if KPIS["expected_margin_approved"] > 0 else "critical"
            kpi_tile("Margin on approved book",
                     money(KPIS["expected_margin_approved"]),
                     "Interest less expected loss", tone)

        left, right = st.columns(2)
        with left:
            section("Expected loss by loan size",
                    "Where is the capital at risk?")
            chart = exposure_chart(BOOK, "amount_band", order=AMOUNT_ORDER)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
        with right:
            section("Expected loss by purpose",
                    "Which product costs the most in losses?")
            chart = exposure_chart(BOOK, "LoanPurpose")
            if chart is not None:
                st.altair_chart(chart, width="stretch")

        section("Largest single exposures",
                "Which individual loans dominate the loss estimate?")
        top = BOOK.sort_values("expected_loss", ascending=False).head(15)
        st.dataframe(pd.DataFrame({
            "Application": top["LoanID"],
            "Decision": top["system_decision"],
            "PD": top["default_probability"],
            "Loan amount": top["LoanAmount"],
            "Expected loss": top["expected_loss"],
            "Primary reason": top["primary_reason"],
        }), width="stretch", hide_index=True, column_config={
            "PD": st.column_config.ProgressColumn("PD", format="%.0f%%",
                                                  min_value=0, max_value=1),
            "Loan amount": st.column_config.NumberColumn(format="$%.0f"),
            "Expected loss": st.column_config.NumberColumn(format="$%.0f"),
        })

    with tabs[2]:
        row = st.columns(3)
        with row[0]:
            kpi_tile("Priced below risk", f"{KPIS['underpriced']:,}",
                     f"{KPIS['underpriced'] / KPIS['total']:.1%} of applications",
                     "critical")
        with row[1]:
            kpi_tile("Exposure underpriced", money(KPIS["underpriced_ead"]),
                     "Amount on those loans", "critical")
        with row[2]:
            kpi_tile("Weighted PD, approved",
                     f"{KPIS['weighted_pd_approved']:.1%}",
                     "Exposure-weighted, auto-approved book", "warning")

        section("Rate charged against rate required",
                "Which loans are priced below their own expected loss?")
        chart = pricing_chart(BOOK)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption("The dashed line is break-even. Points below it are "
                       "loans whose interest income does not cover their "
                       "expected loss, so they lose money in expectation even "
                       "when they perform.")

        section("Underpriced exposure by credit band",
                "Where is the pricing wrong?")
        under = BOOK[BOOK["underpriced"]]
        if under.empty:
            st.info("No applications are priced below their break-even rate.")
        else:
            chart = exposure_chart(under, "credit_band", order=CREDIT_ORDER)
            if chart is not None:
                st.altair_chart(chart, width="stretch")

    with tabs[3]:
        section("Model performance as reported at training",
                "How much should a credit committee trust the score?")
        row = st.columns(4)
        with row[0]:
            kpi_tile("ROC-AUC", f"{CONFIG.get('test_roc_auc', 0):.3f}",
                     "Ranking quality on the held-out test set", "neutral")
        with row[1]:
            kpi_tile("Defaulters caught",
                     f"{CONFIG.get('test_recall_at_threshold', 0):.0%}",
                     "Recall at the operating threshold", "neutral")
        with row[2]:
            kpi_tile("Referral precision",
                     f"{CONFIG.get('test_precision_at_threshold', 0):.0%}",
                     "Share of referrals that do default", "neutral")
        with row[3]:
            kpi_tile("Operating threshold", f"{REFER_THRESHOLD:.0%}",
                     f"Chosen at a {COST_RATIO_MISSED_TO_REJECTED:.0f}:1 cost ratio",
                     "neutral")

        st.caption(
            f"The threshold was selected by minimising expected cost, treating "
            f"a missed default as {COST_RATIO_MISSED_TO_REJECTED:.0f} times as "
            "expensive as a wrongly rejected application. These figures come "
            "from the training notebook and describe the held-out test set, "
            "not the applications loaded here.")

        known = BOOK[BOOK["actual_default"].isin(["Yes", "No"])].copy()
        if known.empty:
            st.info("No applications in this file have a known outcome, so "
                    "performance cannot be re-checked here.")
        else:
            section("Performance on the loaded file",
                    "Does it behave the same on this data?")
            known["actual_binary"] = (known["actual_default"] == "Yes").astype(int)
            known["flagged"] = (known["default_probability"]
                                >= REFER_THRESHOLD).astype(int)
            accuracy = float((known["flagged"] == known["actual_binary"]).mean())
            defaulters = known[known["actual_binary"] == 1]
            recall = (float((defaulters["flagged"] == 1).mean())
                      if len(defaulters) else 0.0)
            flagged = known[known["flagged"] == 1]
            precision = (float((flagged["actual_binary"] == 1).mean())
                         if len(flagged) else 0.0)

            row = st.columns(3)
            with row[0]:
                kpi_tile("Accuracy", f"{accuracy:.1%}",
                         f"On {len(known):,} known outcomes", "neutral")
            with row[1]:
                kpi_tile("Defaulters caught", f"{recall:.1%}",
                         f"{int(defaulters['flagged'].sum()):,} of "
                         f"{len(defaulters):,}", "neutral")
            with row[2]:
                kpi_tile("Referral precision", f"{precision:.1%}",
                         f"{len(flagged):,} flagged", "neutral")
            st.caption("Recall matters more than accuracy here. A model that "
                       "approves everyone scores well on accuracy and catches "
                       "no defaults, which is why the threshold sits below 50%.")

    note = sample_note(BOOK)
    if note:
        st.caption(note)


# ==========================================================================
# System Logs - administrator only
# ==========================================================================

elif page == "System Logs":
    page_header("System Logs", "Application diagnostics and audit trail.")

    connection = get_connection()
    logs = pd.read_sql_query(
        "SELECT timestamp, level, event, LoanID, details FROM system_logs "
        "ORDER BY log_id DESC LIMIT 500", connection)
    connection.close()

    row = st.columns(4)
    with row[0]:
        kpi_tile("Recorded events", f"{len(logs):,}", "Most recent 500",
                 "neutral")
    with row[1]:
        kpi_tile("Errors", f"{int((logs['level'] == 'ERROR').sum()):,}",
                 "Model or data failures", "critical")
    with row[2]:
        kpi_tile("Application file", os.path.basename(SEED_FILE),
                 "Book origin", "neutral")
    with row[3]:
        import sklearn
        kpi_tile("scikit-learn", sklearn.__version__,
                 "Must match the training version", "neutral")

    st.dataframe(logs, width="stretch", hide_index=True, height=440)
    st.caption("Administrator only. The scikit-learn version is shown here "
               "because a mismatch against the version used for training is "
               "the single most common reason the model fails to load.")
