# Trustline-loan-default
# TrustLine

Credit Decisioning & Portfolio Risk

This document explains the whole of `app.py`, top to bottom, in the order the code runs. I wrote it so that someone who does not read Python can still follow what the application does and why. Where a section gets technical I say what the code does in plain language first and put the mechanics after it.

If you only have five minutes, read "What this thing actually does" and "The numbers I quote, and where each one comes from".

## Contents

1. [What this thing actually does](#what-this-thing-actually-does)
2. [How to run it](#how-to-run-it)
3. [The file at a glance](#the-file-at-a-glance)
4. [Settings at the top of the file](#settings-at-the-top-of-the-file)
5. [Finding the model files](#finding-the-model-files)
6. [Theming: how the app handles light and dark](#theming-how-the-app-handles-light-and-dark)
7. [The database](#the-database)
8. [The model layer](#the-model-layer)
9. [Turning a probability into money](#turning-a-probability-into-money)
10. [How I explain a decision](#how-i-explain-a-decision)
11. [Startup: seeding and scoring](#startup-seeding-and-scoring)
12. [The book layer](#the-book-layer)
13. [The charts](#the-charts)
14. [Roles and navigation](#roles-and-navigation)
15. [Page by page](#page-by-page)
16. [The numbers I quote, and where each one comes from](#the-numbers-i-quote-and-where-each-one-comes-from)
17. [What this version cannot do](#what-this-version-cannot-do)
18. [How to change the things you will want to change](#how-to-change-the-things-you-will-want-to-change)

## What this thing actually does

I trained a model that predicts whether a loan applicant will default. A probability on its own is not a lending decision. It does not tell you whether to approve the loan, what rate to charge, whether the rate already on the application covers the risk, who should look at it, or what you would have to tell the applicant if you turned them down.

So the application is built around the sequence a lender actually works through:

```
Score  ->  Price  ->  Decide  ->  Route  ->  Review  ->  Measure
```

Read that as: put a probability on the application, work out what that probability costs in money, decide approve or refer or decline, send the referred ones to a named underwriter, let the underwriter record a final answer, and then count how often the humans disagreed with the model. The model is the first box only.

The decision itself is three-way rather than yes/no:

- **Approve** below a 60% probability of default. It clears without a human.
- **Refer** between 60% and 75%. An underwriter reviews it.
- **Decline** at 75% and above. The model is confident enough that underwriter time is better spent elsewhere.

There are six screens, and which ones you see depends on the role selected in the sidebar.

**Credit Risk Dashboard**: what is being lent, what it is likely to cost, and whether the price covers it.

**Application Queue**: the review worklist, sorted by expected loss rather than by probability.

**Applicant 360**: one application at a time. Look up something already scored, or type in a new application and score it. You get the decision, the pricing, the adverse-action reasons, the sensitivity table, and the buttons that record an underwriting outcome.

**Underwriting Cases**: every case opened, where it sits, how it ended, and whether the underwriter overrode the model.

**Portfolio Analytics**: four tabs answering four questions: who defaults, what it costs, whether the loans are priced for their risk, and whether the model holds up.

**System Logs**: the audit trail, plus the installed scikit-learn version. Administrator only.

## How to run it

Locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens on `http://localhost:8501`.

Deployed, it runs on Streamlit Community Cloud from the `main` branch of this repository, with `app.py` as the main module.

What has to be present:

| File | Why |
|---|---|
| `app.py` | The whole application. One file. |
| `requirements.txt` | Package list, with exact version pins. |
| `trustline_model.joblib` | The trained pipeline: preprocessor and classifier together. |
| `model_config.json` | Threshold, feature order, category options, numeric ranges, test metrics. |
| `data/applications.csv` | The 800 sample applications. |
| `.streamlit/config.toml` | Theme colours and fonts. |

### The pin that cost me a deploy

The first deploy failed with this:

```
AttributeError: Can't get attribute '_RemainderColsList' on
<module 'sklearn.compose._column_transformer'>
```

`requirements.txt` said `scikit-learn>=1.3`. A floor, not a pin. Streamlit Cloud installed the newest release that satisfied it, which was 1.9.0, and `_RemainderColsList` is a private class that exists in 1.6.1 and no longer exists in 1.9.0. My saved pipeline holds a reference to it, because a pickle is a frozen Python object graph rather than a description of one, so unpickling it under the newer library had nothing to bind that name to.

I confirmed it rather than assuming it, by building two virtual environments and importing the class in each. Present in 1.6.1. Gone in 1.9.0.

The line is now `scikit-learn==1.6.1`. Exact, not a floor. `check_versions.py` reads the version string recorded inside the pickle without unpickling it, so you can check what your own model file was actually saved with.

XGBoost is the awkward one. It records no readable version string in the pickle, so nothing can recover it from the file. Loading a model saved by a different XGBoost version produces a warning rather than a failure, which means it will deploy and run, but the warning in the log is exactly that mismatch. The number in `requirements.txt` has to come from the notebook that trained the model:

```python
import xgboost; print(xgboost.__version__)
```

## The file at a glance

`app.py` is about 2,200 lines in a single file. Streamlit Cloud runs one main module, and splitting this across a package would mean managing imports for no benefit at this size. It is divided into labelled blocks separated by banner comments and it runs strictly top to bottom.

| Lines | Block | What it is |
|---|---|---|
| 1–93 | Setup | Imports, product name, risk assumptions, file paths |
| 95–316 | Theming | Two palettes, the stylesheet, small UI helpers |
| 317–540 | Database | Four tables and every read and write |
| 541–863 | Model layer | Loading, scoring, pricing, explaining |
| 865–971 | Startup | Seeding the application file and scoring it once |
| 973–1119 | Book layer | The one table every screen reads from |
| 1120–1288 | Charts | Every chart the app draws |
| 1289–1372 | Roles and navigation | Sidebar, masthead, role permissions |
| 1373–2206 | Pages | The six screens |

One thing to understand before reading further, because it explains most of the design decisions: **Streamlit re-runs this entire file from line 1 every time anyone touches anything.** Move a slider, click a tab, type a character, and the whole script executes again from the top. There are no click handlers the way a conventional web application has them.

That is why the caching matters. Without it, every keystroke would re-read the model from disk and re-score 800 applications.

## Settings at the top of the file

```python
LOSS_GIVEN_DEFAULT = 0.45
DECLINE_THRESHOLD = 0.75
COST_RATIO_MISSED_TO_REJECTED = 5.0
TARGET_MARGIN_BPS = 300
```

Four numbers turn a probability into money. Every one of them is an assumption rather than a measurement, and each is stated on screen wherever it is used.

`LOSS_GIVEN_DEFAULT = 0.45` is the share of principal a lender does not recover when a loan goes bad. Forty-five percent is the Basel foundation-IRB figure for unsecured retail exposure. It is a regulatory default, not a TrustLine recovery rate, and the dashboard says so in the fine print under the tiles.

`DECLINE_THRESHOLD = 0.75` is the line above which an application is declined outright instead of being sent to an underwriter. Referring everything the model flags is only workable while the flagged share stays small enough for a real team to work through.

`COST_RATIO_MISSED_TO_REJECTED = 5.0` records the assumption behind the operating threshold. In the notebook I chose the threshold by minimising expected cost, treating a missed default as five times as expensive as a wrongly rejected application. That asymmetry is why the threshold sits below 50% rather than at it.

`TARGET_MARGIN_BPS = 300` is the margin added on top of break-even when the app suggests a price. Three hundred basis points is three percentage points. It is a required return, not a market rate.

The referral threshold is deliberately not in this list:

```python
try:
    with open(CONFIG_FILE) as _handle:
        REFER_THRESHOLD = float(json.load(_handle).get("threshold", 0.60))
except Exception:
    REFER_THRESHOLD = 0.60
```

It is read out of `model_config.json`, the file the training notebook wrote. If I re-tune the model and the threshold analysis picks a different number, the app follows it without anyone editing Python. The application can never quietly disagree with the analysis that produced it.

## Finding the model files

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(*candidates):
    for candidate in candidates:
        full_path = os.path.join(BASE_DIR, candidate)
        if os.path.exists(full_path):
            return full_path
    return os.path.join(BASE_DIR, candidates[0])
```

`find_file` takes a list of possible locations and returns the first one that exists. The model is looked for next to `app.py`, then in `models/`, then in `model/`. If none of them exist it returns the first path anyway, so the error message on screen names a sensible expected location instead of saying nothing useful.

`BASE_DIR` is the folder `app.py` itself lives in. Everything resolves relative to that rather than to whatever directory the app was launched from, which is what makes the same code work locally and on the server.

The application file uses the same trick:

```python
SEED_FILE = find_file(
    "data/applications.csv",
    "data/Loan_default.csv",
    "Loan_default.csv",
    "applications.csv",
)
```

Drop the full `Loan_default.csv` into `data/` and the app picks it up and scales to it. No code change. The CSV needs the sixteen model columns; `LoanID` and `Default` are optional, and the app generates identifiers and marks outcomes unknown when they are absent.

## Theming: how the app handles light and dark

Streamlit lets each viewer pick light or dark from the settings menu. This app draws a lot of its own HTML for the KPI tiles, the decision banner and the section headings, and hardcoded light text in those blocks would be invisible to anyone viewing in light mode.

There is no CSS-only fix. Streamlit exposes theme variables to custom components but not to the main page, so a stylesheet cannot ask which theme is active. The decision has to be made in Python.

```python
def active_palette():
    try:
        theme_type = st.context.theme.type
    except Exception:
        theme_type = None
    return LIGHT_PALETTE if theme_type == "light" else DARK_PALETTE
```

`st.context.theme.type` reports the theme the viewer is actually looking at. The matching palette is substituted into the stylesheet. Anything unexpected falls back to dark, which is the configured default.

The two palettes hold identical keys with different values: `text`, `heading`, `muted`, `surface`, `border`, chart colours and so on. No colour is hardcoded anywhere else in the file, so changing how the app looks means editing two dictionaries.

`.streamlit/config.toml` carries a `[theme.light]` and a `[theme.dark]` block rather than one shared `[theme]`. That distinction matters. With a single block, the background was pinned to the dark violet while Streamlit still reported the viewer's theme as light, so the Python side chose dark ink and painted it onto a dark ground. Two blocks keep the ground and the ink agreeing with each other.

The stylesheet is a `string.Template` rather than an f-string. CSS is full of curly braces and an f-string would need every one of them doubled. `Template` only treats `$name` as a placeholder, so the CSS stays readable.

### The decision colours are different on purpose

```python
DECISION_COLORS = {"Decline": "#d03b3b", "Refer": "#fab219", "Approve": "#0ca30c"}
DECISION_ICONS = {"Decline": "▲", "Refer": "◆", "Approve": "●"}
```

Red, amber, green, and identical in both themes, unlike everything else in the file. A status colour carries a fixed meaning, so it should not move when the theme does.

The amber has a known weakness: against a near-white background it falls short of the usual contrast standard. Rather than abandon the traffic-light meaning to fix it, every decision indicator carries a shape and a word alongside the colour. The decision-mix chart prints the count and percentage beside each bar. The banner on Applicant 360 spells out DECLINE in text. Nobody has to distinguish the colour to read the screen.

The chart series stayed blue and orange when the app moved to a violet ground. I checked the alternative: on that background, blue and orange separate far better than a green-family pair under colour-vision simulation. The ground changed and the data colours did not.

## The database

Everything is stored in a SQLite file called `trustline_decisions.db`, created on first run. SQLite is a database that lives in one file with no server to install.

Four tables.

**`applications`**: one row per application, holding the sixteen model inputs plus `actual_default` where the outcome is known, `source` (loaded from file, or scored in the app) and `received_at`.

**`scores`**: one row per scoring event. Application reference, the probability, the system decision, the threshold in force at the time, expected loss and expected margin. It is a history rather than a snapshot, so re-scoring an application leaves two rows and the app reads the most recent.

**`underwriting`**: the review queue, and the table that turns this from a report into a product.

```sql
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
```

Two columns carry most of the weight here. `default_probability` and `system_decision` store what the model said *at the moment the case was opened*, not a link to the current score. Six months and one retrain later, you want to know what the system believed when the decision was made. Without that snapshot, `overrode_system` would be meaningless, and the override rate is the one number that tells a credit risk manager whether the human review layer is doing anything.

A case moves through five states: Pending review, In review, and then Approved, Declined or Withdrawn.

**`system_logs`**: timestamped record of seeding, scoring, case changes and any failure to load the model.

### One pattern worth explaining

Nearly every write function looks like this:

```python
def log_event(level, event, loan_id="", details="", connection=None):
    owns = connection is None
    if owns:
        connection = get_connection()

    connection.execute(...)

    if owns:
        connection.commit()
        connection.close()
```

The `connection=None` argument is a performance fix. Every `commit()` forces the operating system to flush to disk, which is slow on a cloud server with network-backed storage. Written the obvious way, seeding 800 applications would open a connection, write one row, commit and close, eight hundred times over.

A caller doing bulk work opens one connection, passes it into every write, and commits once at the end. When nothing is passed the function manages its own connection, so single writes still work unchanged. Seeding and scoring the whole file is two transactions instead of sixteen hundred.

### The double-click guard

```python
def open_case(loan_id, ...):
    existing = get_latest_case(loan_id)
    if existing and existing["status"] not in ("Approved", "Declined", "Withdrawn"):
        return existing["case_id"], False
    ...
    return case_id, True
```

If an application already has a case that has not been closed, the function hands back the existing one instead of creating a second. It reports which of the two happened, so the screen can say "already has an open case" rather than silently doing nothing. I tested it by clicking twice: one case.

## The model layer

`load_artefacts()` reads the pipeline and the config from disk. It carries `@st.cache_resource`, so Streamlit runs it once and hands back the same objects on every later re-run.

It also does something more useful than loading:

```python
except (AttributeError, ModuleNotFoundError, ImportError) as error:
    import sklearn
    raise RuntimeError(
        "The saved model could not be unpickled, which almost always means "
        "the installed library versions differ from the ones the model was "
        "trained with.\n\n"
        f"Installed scikit-learn: {sklearn.__version__}\n"
        ...
    )
```

That is the `_RemainderColsList` failure, caught and rewritten. The raw error names a private class nobody has heard of. This one names the installed version, states the cause, and gives the fix. It cost me a morning to diagnose the first time, and it should cost the next person about ten seconds.

The saved artefact is a scikit-learn `Pipeline` with the preprocessor and the classifier in it, so `model.predict_proba` handles the encoding on its own. `classifier_of()` and `preprocessor_of()` reach inside when the code needs one of the two directly, and both fall back quietly rather than raising if the artefact turns out to be a bare estimator.

`score_frame()` scores any number of applications in one call:

```python
def score_frame(frame, model, feature_order):
    return model.predict_proba(frame[feature_order])[:, 1]
```

The `feature_order` argument comes from the config, and it exists because column order matters to the pipeline. Scoring 800 applications as one table is one call into the model. Scoring them one at a time is eight hundred, and the difference at startup is roughly a second against most of a minute.

`decide()` applies the three-way policy:

```python
def decide(probability):
    if probability >= DECLINE_THRESHOLD:
        return "Decline"
    if probability >= REFER_THRESHOLD:
        return "Refer"
    return "Approve"
```

## Turning a probability into money

This is the part that makes the app a lending tool rather than a scoring demo.

```python
def expected_loss(probability, loan_amount):
    return float(probability) * LOSS_GIVEN_DEFAULT * float(loan_amount)
```

Probability of default times loss given default times exposure at default. PD × LGD × EAD is the standard credit expected-loss identity, and it is the number a credit committee provisions against. It is also why the queue is sorted the way it is: a 70% probability on a $10,000 loan carries an expected loss of about $3,150, while a 45% probability on $200,000 carries $40,500. The second application is worth an underwriter's attention first even though the model thinks it is *less* likely to default.

```python
def expected_interest(loan_amount, interest_rate, loan_term_months):
    return float(loan_amount) * (float(interest_rate) / 100.0) * (
        float(loan_term_months) / 12.0
    )
```

Simple interest over the life of the loan rather than an amortising schedule. On a declining balance the lender earns less than this, so treating it as a ceiling keeps the margin figure conservative in the direction that matters. Overstating income and still showing a loss is a safe error; understating it is not.

```python
def break_even_rate(probability, loan_term_months):
    years = max(float(loan_term_months) / 12.0, 1e-9)
    return float(probability) * LOSS_GIVEN_DEFAULT / years * 100.0
```

The break-even rate is the annual interest rate at which income exactly covers expected loss. Set `expected_interest` equal to `expected_loss` and solve for the rate: the loan amount appears on both sides and cancels, which leaves an answer that depends only on the probability, the loss assumption and the term. That cancellation is the reason a small loan and a large one at the same risk need the same rate.

`price_application()` gathers all of it into one dictionary: expected loss, expected interest, expected margin, break-even rate, a suggested rate at the target margin, risk-adjusted return as a share of principal, and a `priced_correctly` flag that is true when the rate on the application already clears break-even.

That flag drives one of the more interesting findings on the dashboard. An application can be approved by the model and still lose money, because it was approved on risk and priced on something else.

## How I explain a decision

Two different methods, because the two situations have different constraints.

### Across the whole book: the model's own importances

```python
@st.cache_data(show_spinner=False)
def global_drivers(_model, _config):
    classifier = classifier_of(_model)
    preprocessor = preprocessor_of(_model)
    importances = classifier.feature_importances_
    names = list(preprocessor.get_feature_names_out())
```

The classifier reports how much each input mattered, but it reports on the *encoded* columns rather than the human ones. Loan purpose was split during preprocessing into five yes/no columns, one per purpose, so the model returns five numbers where the business wants one.

This function walks every encoded column, matches it back to the application field it came from, and sums the importances. Where two field names could both be prefixes of a column name it keeps the longer match, which stops a short name absorbing importance that belongs to a longer one. The output is the "What drives default" chart.

Those percentages come out of the trained model. I did not write them down and I cannot tune them.

### For one applicant: the sensitivity analysis

```python
def applicant_sensitivity(row, model, config):
    order = config["feature_order"]
    key = tuple((field, row[field]) for field in order)
    return _sensitivity_cached(key, model, config)
```

For a single application the app does something better than importances. It takes the applicant's profile, changes one field, scores them again, and measures how far the probability moved. Numeric fields get shifted a quarter of their range in each direction and clipped to the range from the config. Categorical fields get every other available option. Whichever alternative moves the probability furthest is the one reported.

Do that across all sixteen fields and you learn which single change would help this particular applicant most. That produces the table on Applicant 360, reading something like "Interest rate: currently 22.40, alternative 16.65, change in default probability -4.1%".

Two details in that function are there because of specific problems.

The first is the cache. The analysis costs roughly thirty calls into the model, and Streamlit re-runs the whole script on every interaction, so the page was spending all thirty on every single click. On Streamlit Community Cloud that showed up as a CPU throttle notice on the deployed app. The fix is the wrapper above: it turns the applicant into a tuple, which Streamlit can hash, and hands that to a cached worker. The same applicant is now scored once rather than once per click, and a repeat interaction costs no model calls at all. The `_` prefixes on `_model` and `_config` tell Streamlit not to try hashing those arguments, which it cannot do.

The second is how each modified row is built:

```python
candidate = base_frame.iloc[0].to_dict()
candidate[field] = alternative
modified = pd.DataFrame([candidate])
```

Assigning directly into the frame raised `TypeError: Invalid value '19.25' for dtype 'int64'`. Recent pandas refuses to write a float into an integer column instead of silently upcasting it, which broke on Age, CreditScore and the other whole-number fields. Rebuilding the row from a dictionary lets pandas choose the type fresh, and whole-number fields are rounded before they are used.

### Adverse-action reasons

A lender that declines an application generally has to say why, in specific terms. Running the full sensitivity analysis across a queue of 800 is far too slow for that, so the queue uses a rule ladder.

```python
ADVERSE_RULES = [
    ("Age", lambda r: float(r.get("Age") or 0) < 30,
     "Limited credit history for age"),
    ("InterestRate", lambda r: float(r.get("InterestRate") or 0) > 18,
     "Interest rate above 18%"),
    ("EmploymentType", lambda r: r.get("EmploymentType") == "Unemployed",
     "Applicant not currently employed"),
    ...
]
```

Each rule is a field, a condition and the sentence you would put in a letter. `build_adverse_ladder()` sorts the rules at runtime by the model's global importance for that field, so the ladder is ordered by what the model actually weighs rather than by the order I happened to type them in. `primary_reason()` walks down and returns the first condition the applicant meets; `adverse_reasons()` returns up to four.

The Primary reason column is a rule, not a model output. It is *ordered* by the model, which is what makes it defensible, and it is a fast approximation of the expensive analysis. The one-line version: ranked by the model's own importances, filtered to the conditions this applicant meets.

## Startup: seeding and scoring

Three things happen the first time anyone opens the app.

`seed_applications()` reads the CSV and loads it. It checks first whether the table already has rows, so a restart does not duplicate anything, and it checks that the sixteen model columns are present before touching the database. A missing column produces a logged error rather than a crash halfway through the load. The `Default` column is normalised to Yes, No or Unknown, since the source file stores it as 1 and 0.

`score_pending()` asks the database which applications already have a score, scores only the remainder, and does it in one call. Everything is written inside a single transaction.

`bootstrap()` wraps both:

```python
@st.cache_resource(show_spinner="Scoring the application book...")
def bootstrap():
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
```

`@st.cache_resource` does the heavy lifting. Because Streamlit re-runs the file on every interaction, without it every click would re-seed and re-score the entire book. With it the work happens once per server and every later run gets the cached result instantly.

The two `try` blocks are separate on purpose. A model that will not load stops everything, and the app shows the error, the installed version and the paths it looked in instead of a stack trace. A seed file that fails to load is survivable: the single-applicant scorer still works, and `require_book()` puts a clear message on the screens that need the book rather than letting them fall over.

## The book layer

Every screen reads from one table, built once by `book()`. It joins each application to its most recent score and its most recent case, then adds the columns the business cares about:

- `expected_loss`: probability × 45% × loan amount
- `expected_interest`: rate × amount × term in years
- `expected_margin`: interest less expected loss
- `break_even_rate`: the rate this loan needs to cover its own risk
- `underpriced`: true when the rate charged is below break-even
- `primary_reason`: from the adverse ladder
- `credit_band`, `amount_band`, `age_band`, `rate_band`: for grouping
- `status` and `assigned_to`: from the latest case, or "No case"

The bands exist so the analytics charts have something to group by. Default probability plotted against raw credit score is a cloud of dots. Plotted against five credit bands it answers a question.

The joins use a `MAX(score_id)` and `MAX(case_id)` subquery rather than the plain join a first draft would use, because both `scores` and `underwriting` can hold several rows per application and a plain join would multiply rows instead of picking the newest.

### The caching token

```python
def data_version():
    scores = connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    cases = connection.execute("SELECT COUNT(*) FROM underwriting").fetchone()[0]
    touched = connection.execute(
        "SELECT COALESCE(MAX(updated_at), '') FROM underwriting").fetchone()[0]
    return f"{scores}-{cases}-{touched}"
```

The book needs to be cached so screens are fast, and cached data goes stale. Open a case and the queue would still show the old status.

So the app builds a short string out of the number of scores, the number of cases and the timestamp of the most recent case change, and passes it into the cached function as an argument. Any write changes the string, a changed argument means Streamlit treats it as a new call, and the table rebuilds. When nothing has changed the cache is reused. Three fast counting queries buy the whole cache.

`book_kpis()` turns the book into the numbers the tiles show: approval and referral rates, exposure requested and approved, expected loss on the whole book and on the auto-approved part, exposure-weighted probability, the count of underpriced loans, open cases, and the override rate.

## The charts

Every chart uses Altair, which ships with Streamlit, so there is no extra dependency.

One rule governs the colour: a chart with a single series uses one colour and no legend, because the title already names the series. Colour carries meaning in exactly one chart, the decision mix, where it uses the fixed status palette.

Every chart prints its values directly on the marks. Partly that is the accessibility mitigation for the amber, and mostly it is because an audience watching a presentation cannot hover a tooltip and should not have to estimate a bar against an axis.

`style_chart()` applies the theme to every chart in one place: axis colours, grid colours, transparent background so the app's gradient shows through.

The specific charts:

`risk_rate_chart()`: average default probability within each category. Used for credit band, employment type, age band, loan purpose and rate band.

`exposure_chart()`: expected loss by category. Same grouping, different question. Not where the risky applicants are, but where the money is.

`decision_mix_chart()`: approve, refer and decline counts in the fixed status colours, with counts and percentages printed beside each bar.

`drivers_chart()`: the model's importances aggregated back to application fields, largest first.

`pricing_chart()`: rate charged against break-even rate, with a dashed diagonal. Every point below that line is a loan whose interest income does not cover its own expected loss. This is the chart that usually gets a reaction, because underpriced loans are invisible in any view that only looks at risk. It samples at most 1,500 points with a fixed random seed, so a larger book stays readable and the picture does not shuffle between re-runs.

`sample_note()`: prints an honesty caption when the book holds fewer than 5,000 applications, saying the segment cells are thin and the results are directional.

## Roles and navigation

```python
ROLE_PAGES = {
    "Executive": ["Credit Risk Dashboard", "Portfolio Analytics"],
    "Credit Risk Manager": ["Credit Risk Dashboard", "Application Queue",
                            "Underwriting Cases", "Portfolio Analytics"],
    "Underwriter": ["Application Queue", "Applicant 360", "Underwriting Cases"],
    "Data Analyst": ["Credit Risk Dashboard", "Portfolio Analytics",
                     "Application Queue", "Applicant 360"],
    "Administrator": [everything, including "System Logs"],
}
```

Pick a role in the sidebar and the navigation offers only the pages that role uses. An executive gets the book quality view and the analysis. An underwriter gets their queue and the applicant screens. Only the administrator sees System Logs.

**This demonstrates role-based views, not security.** Anyone can change the dropdown. There is no login and no password, and real credit decisioning would sit behind an identity provider with every decision tied to a named user.

The product point still stands. Different people need different screens, and a chief risk officer should not have to scroll past a system log to find the expected loss on the book.

The sidebar also prints the operating policy under the navigation: refer at 60%, decline at 75%, LGD 45%. Those are the three numbers everything else depends on, and they are visible on every screen rather than buried in the code.

## Page by page

### Credit Risk Dashboard

Six tiles across two rows: applications scored, auto-approval rate, referrals to underwriting, amount requested, expected loss, and loans priced below risk.

Under them, fine print spelling out that expected loss is PD × LGD × EAD and that the 45% is a Basel assumption rather than a measured recovery rate. Someone will ask, and the answer should be on the screen rather than in the presenter's head.

Below that, a book composition warning that appears only when the known default rate in the loaded file differs from the 11.6% training population by more than six percentage points. With the current sample it stays hidden, which is the point: the 800 applications were drawn stratified on the outcome, so the sample defaults at 11.25% against 11.6% in the full population and the rates on screen describe something close to the real book.

Then four charts (decision mix, default risk by credit band, what drives default, expected loss by loan size) and a table of the ten largest exposures with no case open yet, sorted by expected loss.

### Application Queue

Filters across the top: system decision, case status, sort field, and a reference search. It opens showing Decline and Refer only, since those are the applications that need a person. Four summary tiles recalculate for whatever the filters currently show.

The worklist columns are: Application, Default probability, Decision, Loan amount, Expected loss, Rate charged, Break-even rate, Primary reason, Status, Owner. The probability renders as a progress bar rather than a number, so the column can be scanned instead of read.

Putting rate charged and break-even rate side by side is the whole idea of the screen. An underwriter can see, in one row, both whether the applicant is risky and whether the price already accounts for it.

At the bottom, pick an application, an underwriter and an opening status, and open a case.

### Applicant 360

Two tabs. One looks up an application already in the book, ordered by expected loss. The other takes a fresh application through a form of sixteen inputs arranged in three columns under Applicant, Finances and The loan. Every slider's range and default comes from `model_config.json` rather than being typed in, so the form can never offer a value the model was not trained across.

The submit button sits at the top beside the application reference rather than below all sixteen fields.

Either route arrives at the same result, drawn by `render_decision()`:

A coloured banner with the verdict, the reference and the probability at large size. A caption comparing the probability to the book average, so a bare percentage becomes "2.1 times the average applicant". Four figures: loan amount, expected loss, expected interest, expected margin. A pricing verdict that names the break-even rate, the rate charged and, where the loan is underpriced, the rate it would need at the target margin. The adverse factors. The sensitivity table. Then four buttons that write to the underwriting table: send to underwriting, approve, decline, request more information.

Approve and Decline both set `overrode_system` when the human answer disagrees with the model. I tested that path end to end: approving an application the system had declined produced a case with the override flag set, and the override rate on the Underwriting Cases screen moved to match.

One implementation detail worth remembering: Streamlit renders both tabs even when only one is visible, so `render_decision` runs twice on every load. Two buttons with the same identifier crash the app. The function takes a `key_prefix` and the two tabs pass different values.

### Underwriting Cases

Four tiles: cases opened, in the queue, decided, and the override rate.

The override rate is the most useful number on the screen, and the fine print under it explains why. An override is an underwriter approving what the model would refer or decline, or declining what it would approve. A rate near zero means the review layer is rubber-stamping and adds nothing. A very high rate means the thresholds are set wrong. The number is a check on both the model and the process.

Below that, a pipeline chart of how many cases sit at each stage, the full case book showing the probability and system decision as they stood when each case opened, and a form to update a case. Setting the status to Approved or Declined sets the final decision and recomputes the override flag.

### Portfolio Analytics

Four tabs, each named for the question it answers.

**Who defaults**: default risk by credit band, employment type, age band, loan purpose and rate band. The rate band chart carries a caption noting that a rising line there is partly circular, since the lender already charges risky applicants more, which makes rate both a cause and a symptom.

**What it costs**: expected loss on the whole book against the auto-approved part, margin on the approved book, expected loss by loan size and by purpose, and the fifteen individual applications that dominate the loss estimate.

**Are we pricing it**: how many loans sit below break-even and how much exposure that represents, the exposure-weighted probability on the approved book, the pricing scatter, and underpriced exposure by credit band.

**Does the model hold up**: ROC-AUC 0.759, recall 54.4% and precision 28.2% at the operating threshold, all read from the config the notebook wrote, plus the same three metrics recomputed on whatever applications in the loaded file have a known outcome. Recall matters more than accuracy here. A model that approves everyone scores 88% on accuracy and catches nothing.

Every chart has a caption naming the business question it answers. A chart that does not answer a question is decoration.

### System Logs

Event count, error count, which application file the book came from, and the installed scikit-learn version, then the last 500 log lines. Administrator only.

The version tile is there deliberately. A mismatch against the training version is the single most common reason this app fails to load, and it took a deploy failure to learn that, so the number is now one click away instead of buried in a build log.

## The numbers I quote, and where each one comes from

| Number | Formula | Watch out for |
|---|---|---|
| Applications scored | Row count in `applications` | 800 today. Scales to whatever CSV is present. |
| Auto-approval rate | Applications below 60% ÷ total | Uses the config threshold, not a hardcoded one. |
| Referred to underwriting | Count between 60% and 75% | The workload the policy creates. |
| Amount requested | Sum of loan amounts | Exposure at default if every application were written. |
| Expected loss | Σ (PD × 0.45 × amount) | The 45% is a Basel assumption, not a measured recovery rate. |
| Expected interest | amount × rate × term in years | Simple interest, so it overstates income on an amortising loan. |
| Expected margin | Expected interest less expected loss | Negative means the loan loses money in expectation. |
| Break-even rate | PD × 0.45 ÷ term in years, as a percent | Independent of loan size; the amount cancels. |
| Suggested rate | Break-even plus 300 basis points | A required margin, not a market price. |
| Priced below risk | Rate charged < break-even rate | Can be true on an approved application. |
| Risk-adjusted return | Expected margin ÷ loan amount | Return per dollar of principal at risk. |
| Weighted PD, approved | Σ (PD × amount) ÷ Σ amount on approvals | Exposure-weighted, so large loans count more. |
| Override rate | Overrides ÷ decided cases | Undefined until at least one case is decided. |
| ROC-AUC, recall, precision | From `model_config.json` | Measured on the held-out test set at training, not on this file. |
| Accuracy on the loaded file | Correct flags ÷ known outcomes | Only applications with a known outcome count. |

Two of these are worth keeping straight. Expected loss on the whole book assumes every application is written, including the ones the model declines. Expected loss on the auto-approved book is what actually clears without a human. The gap between them is the value the decision policy is adding, and it is the more honest figure to plan against.

## What this version cannot do

**The loss given default is an assumption.** Forty-five percent is a regulatory default for unsecured retail lending, and a real lender would replace it with its own recovery experience. Every money figure in the app scales linearly with it, so halving it halves every expected loss on every screen.

**The book is 800 applications.** They were sampled stratified on the outcome, so the default rate matches the full population closely, but segment cells are thin and the app says so in a caption below the analytics. Dropping the full `Loan_default.csv` into `data/` fixes it with no code change.

**The database resets.** Streamlit Community Cloud gives each app a temporary filesystem, so cases opened during a demo survive until the container restarts and then the app re-seeds from the CSV. Fine for a presentation, not fine for real use. Production would need a hosted database, and credit decisions are exactly the kind of record that must outlive a container.

**Role switching is not security.** Covered above. It shapes what the product shows and protects nothing.

**Primary reason in the queue is a rule, not a model output.** The rules are ordered by the model's importances, but the per-applicant analysis only runs on Applicant 360.

**Expected interest is simple, not amortising.** A real schedule would show less income, so the margin figures here are the optimistic end of the range.

**There is no time series.** Every application in the file was scored in the same instant, so the app cannot show risk trending over weeks. The age band and rate band charts answer a similar question from the data that exists.

**The model itself is moderate.** ROC-AUC 0.759 and 54.4% recall at the operating threshold means it catches about half of the defaults it sees. The app is built to be honest about that rather than to hide it, which is why the referral layer exists at all: the model narrows the queue, a person decides the hard cases, and the override rate measures whether that division of labour is working.

## How to change the things you will want to change

**Change the operating threshold.** Edit `threshold` in `model_config.json`. Everything downstream follows: the decisions, the tiles, the queue, the charts.

**Change the decline cut-off.** `DECLINE_THRESHOLD` at the top of `app.py`.

**Change the loss assumption.** `LOSS_GIVEN_DEFAULT`. Every money figure in the app moves with it.

**Change the target margin.** `TARGET_MARGIN_BPS`, in basis points.

**Change the colours.** Edit `DARK_PALETTE` and `LIGHT_PALETTE`. Both hold the same keys, and no colour is hardcoded anywhere else.

**Change the fonts or the default theme.** That is `.streamlit/config.toml`, not `app.py`.

**Add an underwriter or a decline reason.** Append to `UNDERWRITERS` or `DECLINE_REASONS` near the top of the database section. Both dropdowns read from those lists.

**Add an adverse-action reason.** Append a tuple to `ADVERSE_RULES` with the field it relates to, a condition, and the sentence. The ladder sorts itself by model importance.

**Add a chart.** Write a function that returns an Altair chart, wrap it in `style_chart()`, and call it from a page. `risk_rate_chart` is the simplest one to copy.

**Add a page.** Add an `elif page == "..."` branch at the bottom, then add the page name to whichever roles should see it in `ROLE_PAGES`.

**Load the full dataset.** Put the CSV at `data/Loan_default.csv` and delete `trustline_decisions.db` if you are running locally. Everything scales on its own.

**Re-train the model.** Save the new pipeline as `trustline_model.joblib`, write the new `model_config.json` from the notebook, and update the `scikit-learn` and `xgboost` pins in `requirements.txt` to the versions you trained with. Those have to match, and the pin is the part that is easy to forget.
