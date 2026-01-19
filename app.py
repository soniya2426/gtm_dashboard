import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import statsmodels.api as sm
from scipy import stats


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="IOTA Water UAE | GTM Analytics Dashboard",
    layout="wide",
)

st.title("IOTA Water UAE | GTM Analytics Dashboard")
st.caption(
    "Decision-grade consumer research analytics for founders, strategy teams, and investors. "
    "Built around STP (Segmentation, Targeting, Positioning)."
)

# =========================
# GitHub data config
# =========================
# Replace these with your GitHub username/repo/branch after you upload your CSV.
DEFAULT_GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/<YOUR_BRANCH>/"
    "data/Research_for_bottled_water_UAE_200_respondents.csv"
)

# =========================
# Helpers
# =========================
def standardize_colname(col: str) -> str:
    """Make column names consistent: snake_case, remove punctuation, trim."""
    col = str(col).strip().lower()
    col = re.sub(r"[^\w\s]", " ", col)     # replace punctuation with spaces
    col = re.sub(r"\s+", "_", col)         # spaces -> underscores
    col = re.sub(r"_+", "_", col)          # multiple underscores -> one
    col = col.strip("_")
    return col


def clean_brand_list_text(x: str) -> List[str]:
    """Split comma-separated brand strings into a clean list."""
    if pd.isna(x):
        return []
    s = str(x)
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    return parts


def mode_impute(series: pd.Series):
    """Impute categorical with mode (most frequent)."""
    if series.dropna().empty:
        return series
    return series.fillna(series.dropna().mode().iloc[0])


def median_impute(series: pd.Series):
    """Impute numeric with median."""
    if series.dropna().empty:
        return series
    return series.fillna(series.median())


def safe_value_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=False).reset_index()
    vc.columns = [col, "count"]
    vc["share"] = (vc["count"] / vc["count"].sum()).round(3)
    return vc


def map_monthly_spend_to_aed(spend: str) -> float:
    """
    Map your categorical monthly spend to numeric AED midpoint.
    Adjust ranges here if your survey categories change.
    """
    if pd.isna(spend):
        return np.nan
    s = str(spend).strip().lower()

    # Common patterns seen in your file
    if "below" in s and "50" in s:
        return 25.0
    if "50" in s and ("100" in s or "99" in s):
        return 75.0
    if "100" in s and "200" in s:
        return 150.0
    if "200" in s and "300" in s:
        return 250.0
    if "300" in s and "500" in s:
        return 400.0
    if "500" in s or "above" in s:
        return 600.0

    # Fallback: try to extract numbers and take midpoint
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        a, b = float(nums[0]), float(nums[1])
        return (a + b) / 2
    if len(nums) == 1:
        return float(nums[0])

    return np.nan


def map_frequency_to_score(x: str) -> float:
    """Ordinal scoring for purchase frequency."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()

    mapping = {
        "more than once a week": 5,
        "once a week": 4,
        "once a month": 2,
        "rarely": 1,
        "never": 0,
    }
    # fuzzy match
    for k, v in mapping.items():
        if k in s:
            return float(v)
    return np.nan


def map_eatout_freq_to_score(x: str) -> float:
    """Ordinal scoring for eat-out frequency."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()

    mapping = {
        "more than once a week": 5,
        "once a week": 4,
        "once a fortnite": 3,  # typo preserved from survey
        "once a fortnight": 3,
        "once a month": 2,
        "rarely": 1,
        "never": 0,
    }
    for k, v in mapping.items():
        if k in s:
            return float(v)
    return np.nan


def yes_no_to_binary(x: str) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["yes", "y", "true", "1"]:
        return 1.0
    if s in ["no", "n", "false", "0"]:
        return 0.0
    return np.nan


def pick_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first candidate column that exists."""
    for c in candidates:
        if c in cols:
            return c
    return None


# =========================
# Load data
# =========================
@st.cache_data(show_spinner=False)
def load_data_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Data Overview",
            "Consumer Insights",
            "Hypothesis Testing & Modeling",
            "Segmentation (STP)",
            "Positioning & Perceptual Maps",
        ],
    )

    st.divider()
    st.subheader("Data source")
    data_url = st.text_input("GitHub Raw CSV URL", value=DEFAULT_GITHUB_RAW_URL)
    uploaded = st.file_uploader("Or upload CSV (fallback)", type=["csv"])
    st.caption(
        "Best practice: load from GitHub Raw URL for Streamlit Cloud deployments. "
        "Upload is a safe fallback for local testing."
    )


# Try URL first, else upload
df_raw = None
load_error = None

try:
    if data_url and "<YOUR_GITHUB_USERNAME>" not in data_url:
        df_raw = load_data_from_url(data_url)
except Exception as e:
    load_error = str(e)

if df_raw is None and uploaded is not None:
    try:
        df_raw = load_data_from_upload(uploaded)
        load_error = None
    except Exception as e:
        load_error = str(e)

if df_raw is None:
    st.warning(
        "Data not loaded yet. Paste your GitHub Raw CSV URL (recommended), "
        "or upload the CSV in the sidebar."
    )
    if load_error:
        st.info(f"Load error detail (helpful for debugging): {load_error}")
    st.stop()


# =========================
# Cleaning + feature engineering
# =========================
df = df_raw.copy()

# Drop obvious junk columns (common in Excel exports)
junk_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
if junk_cols:
    df = df.drop(columns=junk_cols, errors="ignore")

# Standardize column names
df.columns = [standardize_colname(c) for c in df.columns]

# Identify key columns (based on your dataset)
cols = df.columns.tolist()

col_age = pick_existing(cols, ["please_specify_your_age", "age"])
col_work = pick_existing(cols, ["what_is_your_current_work_status", "work_status"])
col_nationality = pick_existing(cols, ["what_is_your_nationality", "nationality"])

col_purchase_freq = pick_existing(cols, ["how_often_do_you_purchase_packaged_drinking_water"])
col_purchase_channel = pick_existing(cols, ["where_do_you_usually_buy_bottled_water"])
col_pack_size = pick_existing(cols, ["what_size_of_bottled_water_do_you_buy_most_often"])
col_monthly_spend = pick_existing(cols, ["what_is_your_average_monthly_spent_on_water_in_a_month"])

col_awareness = pick_existing(cols, ["which_brands_are_you_aware_of"])
col_most_freq_brand = pick_existing(cols, ["which_brands_of_bottled_water_do_you_purchase_most_frequently"])
col_reason = pick_existing(cols, ["based_on_the_above_question_please_specify_the_reason_for_this_preference"])

col_eatout_freq = pick_existing(cols, ["how_often_do_you_eat_out"])
col_buy_water_eatout = pick_existing(cols, ["do_you_buy_water_while_eating_out"])
col_type_eatout = pick_existing(cols, ["which_one_of_the_following_you_buy_when_eating_out"])
col_restaurant_brands = pick_existing(cols, ["what_brands_do_you_usually_buy_at_the_restaurant"])

# Importance ratings (numeric 1-5 in your file)
importance_candidates = [
    "how_important_is_value_for_money_in_purchasing_bottled_water",
    "how_important_is_packaging_type_in_purchasing_bottled_water",
    "how_important_is_added_benefits_like_alkaline_zero_sodium_added_minerals_in_purchasing_bottled_water",
    "how_important_is_source_of_water_in_purchasing_bottled_water",
    "how_important_is_availability_in_purchasing_bottled_water",
    "how_important_is_taste_in_purchasing_bottled_water",
    "how_important_is_brand_name_in_purchasing_bottled_water",
    "how_important_is_attractive_promotions_in_purchasing_bottled_water",
]
importance_cols = [c for c in importance_candidates if c in cols]

# Convert importance columns to numeric (robust)
for c in importance_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Feature engineering: numeric WTP proxy + behavior scores
if col_monthly_spend:
    df["monthly_spend_aed"] = df[col_monthly_spend].apply(map_monthly_spend_to_aed)
else:
    df["monthly_spend_aed"] = np.nan

if col_purchase_freq:
    df["purchase_freq_score"] = df[col_purchase_freq].apply(map_frequency_to_score)
else:
    df["purchase_freq_score"] = np.nan

if col_eatout_freq:
    df["eatout_freq_score"] = df[col_eatout_freq].apply(map_eatout_freq_to_score)
else:
    df["eatout_freq_score"] = np.nan

if col_buy_water_eatout:
    df["buys_water_when_eating_out"] = df[col_buy_water_eatout].apply(yes_no_to_binary)
else:
    df["buys_water_when_eating_out"] = np.nan

# Missing value handling:
# - numeric -> median
# - categorical -> mode
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

for c in num_cols:
    df[c] = median_impute(df[c])

for c in cat_cols:
    df[c] = mode_impute(df[c])


# =========================
# Global filters (executive-friendly)
# =========================
with st.sidebar:
    st.divider()
    st.subheader("Filters")
    # Only show filters for columns that exist
    if col_age:
        age_options = sorted(df[col_age].astype(str).unique().tolist())
        age_filter = st.multiselect("Age", options=age_options, default=age_options)
    else:
        age_filter = None

    if col_nationality:
        nat_options = sorted(df[col_nationality].astype(str).unique().tolist())
        nat_filter = st.multiselect("Nationality", options=nat_options, default=nat_options)
    else:
        nat_filter = None

    if col_purchase_channel:
        ch_options = sorted(df[col_purchase_channel].astype(str).unique().tolist())
        ch_filter = st.multiselect("Purchase Channel", options=ch_options, default=ch_options)
    else:
        ch_filter = None

def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if col_age and age_filter is not None:
        out = out[out[col_age].astype(str).isin(age_filter)]
    if col_nationality and nat_filter is not None:
        out = out[out[col_nationality].astype(str).isin(nat_filter)]
    if col_purchase_channel and ch_filter is not None:
        out = out[out[col_purchase_channel].astype(str).isin(ch_filter)]
    return out

df_f = apply_filters(df)


# =========================
# Page: Data Overview
# =========================
if page == "Data Overview":
    st.subheader("Data Health & Quality Checks")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df_f.shape[0]:,}")
    c2.metric("Columns", f"{df_f.shape[1]:,}")
    c3.metric("Numeric features", f"{len(num_cols):,}")

    with st.expander("Dataset shape, dtypes, missing value summary", expanded=True):
        colA, colB = st.columns(2)

        # Missing values summary (post-imputation will be mostly zero; show raw too if possible)
        missing = df_raw.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].reset_index()
        missing.columns = ["column", "missing_count"]

        colA.write("**Data types (cleaned)**")
        dtypes_df = pd.DataFrame({"column": df_f.columns, "dtype": [str(t) for t in df_f.dtypes]})
        colA.dataframe(dtypes_df, use_container_width=True, height=360)

        colB.write("**Missing values (raw, before imputation)**")
        if missing.empty:
            colB.success("No missing values detected in the raw file.")
        else:
            colB.dataframe(missing, use_container_width=True, height=360)

    st.divider()
    st.subheader("Preview")
    st.dataframe(df_f.head(25), use_container_width=True)

    st.info(
        "So what? This page tells you if the survey data is trustworthy enough to steer GTM decisions. "
        "If missingness is high in any key variable, treat those insights as directional, not gospel."
    )


# =========================
# Page: Consumer Insights (>= 10 outputs live here)
# =========================
elif page == "Consumer Insights":
    st.subheader("Consumer Insights (Descriptive + Diagnostic)")

    tab1, tab2, tab3 = st.tabs(["Demographics", "Behavior", "Brands & Drivers"])

    # ---- Demographics ----
    with tab1:
        st.markdown("### Demographic distribution")
        cols_to_plot = []
        if col_age: cols_to_plot.append(col_age)
        if col_work: cols_to_plot.append(col_work)
        if col_nationality: cols_to_plot.append(col_nationality)

        if not cols_to_plot:
            st.warning("No demographic columns detected.")
        else:
            for colx in cols_to_plot:
                vc = safe_value_counts(df_f, colx)
                fig = px.bar(vc, x=colx, y="count", text="share")
                fig.update_layout(height=380, yaxis_title="Respondents")
                st.plotly_chart(fig, use_container_width=True)

                if colx == col_age:
                    st.caption(
                        "So what? Your age mix signals whether IOTA should skew premium (young professionals) "
                        "or mass (family households). GTM implication: tailor channel and messaging accordingly."
                    )
                elif colx == col_work:
                    st.caption(
                        "So what? Work status is a proxy for lifestyle and routine. "
                        "GTM implication: office-heavy segments often over-index on convenience and multi-pack buying."
                    )
                elif colx == col_nationality:
                    st.caption(
                        "So what? UAE is a multi-cultural market. "
                        "GTM implication: bilingual packaging + culturally neutral value props help scale faster."
                    )

    # ---- Behavior ----
    with tab2:
        st.markdown("### Consumption behavior and purchase mechanics")

        plot_cols = [
            (col_purchase_freq, "Purchase frequency"),
            (col_pack_size, "Preferred pack size"),
            (col_purchase_channel, "Preferred channel"),
            (col_eatout_freq, "Eat-out frequency"),
            (col_buy_water_eatout, "Buys water when eating out"),
            (col_type_eatout, "Type bought when eating out"),
        ]

        for c, title in plot_cols:
            if c:
                vc = safe_value_counts(df_f, c)
                fig = px.bar(vc, x=c, y="count", text="share", title=title)
                fig.update_layout(height=360, title_font_size=16, yaxis_title="Respondents")
                st.plotly_chart(fig, use_container_width=True)

                if c == col_purchase_channel:
                    st.caption(
                        "So what? Channel preference is your distribution strategy in disguise. "
                        "GTM implication: if supermarkets dominate, win shelf + visibility; if online/home delivery matters, "
                        "bundle subscriptions and bulk packs."
                    )
                if c == col_pack_size:
                    st.caption(
                        "So what? Pack size drives unit economics and positioning (single-serve premium vs bulk utility). "
                        "GTM implication: align packaging formats with your highest-value segment and channel."
                    )

        # Diagnostic: Cross-tab
        st.markdown("### Diagnostic cross-tabs (who behaves differently?)")

        ct_col1, ct_col2 = st.columns(2)

        # Cross-tab 1: Age vs monthly spend (WTP proxy)
        if col_age and col_monthly_spend:
            ctab = pd.crosstab(df_f[col_age].astype(str), df_f[col_monthly_spend].astype(str), normalize="index")
            fig = px.imshow(ctab, text_auto=".2f", aspect="auto", title="Age vs Monthly Spend (share within age)")
            fig.update_layout(height=420)
            ct_col1.plotly_chart(fig, use_container_width=True)
            ct_col1.caption(
                "So what? This shows which ages over-index on higher monthly spend (proxy for willingness to pay). "
                "GTM implication: prioritize premium positioning where higher spend is concentrated."
            )
        else:
            ct_col1.info("Cross-tab Age vs Monthly Spend not available (missing columns).")

        # Cross-tab 2: Channel vs pack size
        if col_purchase_channel and col_pack_size:
            ctab2 = pd.crosstab(df_f[col_purchase_channel].astype(str), df_f[col_pack_size].astype(str), normalize="index")
            fig2 = px.imshow(ctab2, text_auto=".2f", aspect="auto", title="Channel vs Pack Size (share within channel)")
            fig2.update_layout(height=420)
            ct_col2.plotly_chart(fig2, use_container_width=True)
            ct_col2.caption(
                "So what? Some pack sizes win in some channels (e.g., bulk online, single-serve convenience). "
                "GTM implication: don’t launch every SKU everywhere; match SKUs to channel economics."
            )
        else:
            ct_col2.info("Cross-tab Channel vs Pack Size not available (missing columns).")

        # Correlation heatmap across numeric perceptions and behavior
        st.markdown("### Correlation heatmap (perceptions and behavior)")
        corr_features = []
        corr_features += importance_cols
        for extra in ["monthly_spend_aed", "purchase_freq_score", "eatout_freq_score", "buys_water_when_eating_out"]:
            if extra in df_f.columns:
                corr_features.append(extra)

        if len(corr_features) >= 3:
            corr = df_f[corr_features].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "So what? Correlations tell you what tends to move together (not what causes what). "
                "GTM implication: if price-sensitivity proxies correlate strongly with spend/WTP, pricing architecture matters."
            )
        else:
            st.info("Not enough numeric features detected to compute a correlation heatmap.")

    # ---- Brands & Drivers ----
    with tab3:
        st.markdown("### Brand awareness and brand purchased most frequently")

        c1, c2 = st.columns(2)

        # Awareness: explode lists
        if col_awareness:
            all_awareness = []
            for x in df_f[col_awareness].astype(str).tolist():
                all_awareness.extend(clean_brand_list_text(x))
            awareness_counts = pd.Series(all_awareness).value_counts().reset_index()
            awareness_counts.columns = ["brand", "mentions"]
            awareness_counts = awareness_counts.head(15)

            fig = px.bar(awareness_counts, x="brand", y="mentions", title="Top brand awareness (mentions)")
            fig.update_layout(height=420, xaxis_title="Brand", yaxis_title="Mentions")
            c1.plotly_chart(fig, use_container_width=True)
            c1.caption(
                "So what? Awareness shows who owns mindshare. "
                "GTM implication: if incumbents dominate, IOTA needs a sharp wedge (health, taste, packaging, or channel)."
            )
        else:
            c1.info("Brand awareness column not found.")

        if col_most_freq_brand:
            vc = safe_value_counts(df_f, col_most_freq_brand).head(15)
            fig = px.bar(vc, x=col_most_freq_brand, y="count", title="Most frequently purchased brand")
            fig.update_layout(height=420, xaxis_title="Brand", yaxis_title="Respondents")
            c2.plotly_chart(fig, use_container_width=True)
            c2.caption(
                "So what? ‘Most purchased’ is the competitive reality check. "
                "GTM implication: this is who IOTA must beat at shelf, online search, and restaurant listings."
            )
        else:
            c2.info("Most frequently purchased brand column not found.")

        st.markdown("### Purchase drivers (importance ratings 1–5)")
        if importance_cols:
            imp_means = df_f[importance_cols].mean().sort_values(ascending=False).reset_index()
            imp_means.columns = ["driver", "avg_importance"]
            imp_means["driver"] = imp_means["driver"].str.replace("_", " ").str.title()

            fig = px.bar(imp_means, x="avg_importance", y="driver", orientation="h", title="Average importance by driver")
            fig.update_layout(height=520, xaxis_title="Avg rating (1–5)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "So what? These are the category-level decision criteria. "
                "GTM implication: make your top 2–3 drivers painfully obvious in packaging + ads + PDP (product detail pages)."
            )
        else:
            st.info("Importance rating columns not found. Check your survey export.")


# =========================
# Page: Hypothesis Testing & Modeling
# =========================
elif page == "Hypothesis Testing & Modeling":
    st.subheader("Hypothesis Testing & Linear Regression (WTP proxy)")

    st.markdown(
        "Your dataset does not include a direct 'purchase intent' scale. "
        "So the dashboard uses **Monthly Spend (AED midpoint)** as a practical proxy for **willingness to pay / wallet share**. "
        "This is common in GTM work when intent isn’t measured."
    )

    # Define model variables
    y_col = "monthly_spend_aed"

    # Pick predictors from importance drivers + convenience proxy + promo + brand name etc.
    X_candidates = []
    X_candidates += importance_cols
    for extra in ["purchase_freq_score", "eatout_freq_score", "buys_water_when_eating_out"]:
        if extra in df_f.columns:
            X_candidates.append(extra)

    # Remove any obvious leakage or nonsense
    X_candidates = list(dict.fromkeys([c for c in X_candidates if c != y_col]))

    if y_col not in df_f.columns or df_f[y_col].isna().all():
        st.error("Monthly spend proxy is not available. Check the monthly spend column mapping.")
        st.stop()

    if len(X_candidates) < 3:
        st.error("Not enough predictors available to run regression. Check that importance rating columns exist.")
        st.stop()

    # Build model dataframe
    model_df = df_f[[y_col] + X_candidates].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    # OLS regression
    X = model_df[X_candidates]
    y = model_df[y_col]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        st.markdown("### Regression results (OLS)")
        results_table = pd.DataFrame(
            {
                "feature": model.params.index,
                "coef": model.params.values,
                "p_value": model.pvalues.values,
            }
        )
        results_table["abs_coef"] = results_table["coef"].abs()
        results_table = results_table.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
        st.dataframe(results_table, use_container_width=True, height=420)

        st.metric("R-squared", f"{model.rsquared:.3f}")

        st.caption(
            "So what? Coefficients show which perceptions/behaviors are most associated with higher monthly spend. "
            "GTM implication: invest messaging and product design into the strongest (and significant) drivers."
        )

    with c2:
        st.markdown("### Hypotheses (accept/reject)")
        st.caption("Rule: p < 0.05 ⇒ statistically supported (direction still matters).")

        # Define hypotheses tied to your real columns
        # (You can rename these easily later.)
        hypotheses = []

        def add_hypothesis(label: str, feature_col: str, expected_sign: str):
            if feature_col in model.params.index:
                coef = model.params[feature_col]
                p = model.pvalues[feature_col]
                direction_ok = (coef > 0) if expected_sign == "+" else (coef < 0)
                supported = (p < 0.05) and direction_ok
                hypotheses.append(
                    {
                        "hypothesis": label,
                        "feature": feature_col,
                        "expected": expected_sign,
                        "coef": coef,
                        "p_value": p,
                        "decision": "ACCEPT" if supported else "REJECT",
                    }
                )

        # Map to likely GTM hypotheses using your importance drivers
        add_hypothesis("H1: Higher importance on taste is linked to higher spend (premium inclination).",
                       "how_important_is_taste_in_purchasing_bottled_water", "+")
        add_hypothesis("H2: Higher importance on source is linked to higher spend (quality cue).",
                       "how_important_is_source_of_water_in_purchasing_bottled_water", "+")
        add_hypothesis("H3: Higher importance on added benefits is linked to higher spend (functional premium).",
                       "how_important_is_added_benefits_like_alkaline_zero_sodium_added_minerals_in_purchasing_bottled_water", "+")
        add_hypothesis("H4: Higher price-sensitivity (value for money importance) is linked to lower spend (price pressure).",
                       "how_important_is_value_for_money_in_purchasing_bottled_water", "-")
        add_hypothesis("H5: Higher importance on availability is linked to higher spend (convenience buyers spend more).",
                       "how_important_is_availability_in_purchasing_bottled_water", "+")

        hyp_df = pd.DataFrame(hypotheses)
        if hyp_df.empty:
            st.info("No hypothesis features found in the model. Check column detection.")
        else:
            show_cols = ["hypothesis", "expected", "coef", "p_value", "decision"]
            st.dataframe(hyp_df[show_cols], use_container_width=True, height=420)

        st.caption(
            "So what? This turns survey data into ‘investor-grade’ claims. "
            "GTM implication: only scale the story that the data supports."
        )


# =========================
# Page: Segmentation (STP)
# =========================
elif page == "Segmentation (STP)":
    st.subheader("Segmentation (Data-driven) → Targeting Recommendations")

    st.markdown(
        "This segmentation uses **K-Means clustering** over standardized features: "
        "purchase frequency, monthly spend (WTP proxy), eat-out behavior, and importance drivers."
    )

    # Segmentation features
    seg_features = []
    seg_features += importance_cols
    for extra in ["monthly_spend_aed", "purchase_freq_score", "eatout_freq_score", "buys_water_when_eating_out"]:
        if extra in df_f.columns:
            seg_features.append(extra)

    seg_features = [c for c in seg_features if c in df_f.columns]

    if len(seg_features) < 4:
        st.error("Not enough features to build a stable segmentation. Ensure importance ratings + spend exist.")
        st.stop()

    seg_df = df_f[seg_features].copy()
    seg_df = seg_df.replace([np.inf, -np.inf], np.nan).dropna()

    with st.sidebar:
        st.divider()
        st.subheader("Segmentation settings")
        k = st.slider("Number of segments (K)", min_value=3, max_value=6, value=4)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(seg_df)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(Xs)

    seg_out = seg_df.copy()
    seg_out["segment_id"] = clusters

    # Profile segments
    profile = seg_out.groupby("segment_id")[seg_features].mean()
    sizes = seg_out["segment_id"].value_counts().sort_index()
    profile["size"] = sizes
    profile["share"] = (profile["size"] / profile["size"].sum()).round(3)

    st.markdown("### Segment sizes")
    size_df = pd.DataFrame({"segment_id": sizes.index, "size": sizes.values})
    fig = px.bar(size_df, x="segment_id", y="size", title="Segment size (respondents)")
    fig.update_layout(height=360, xaxis_title="Segment", yaxis_title="Respondents")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "So what? Size tells you where scale lives. "
        "GTM implication: don’t over-invest in a tiny segment unless margins are huge."
    )

    st.markdown("### Segment profiles (average values)")
    st.dataframe(profile.reset_index(), use_container_width=True, height=420)

    # Give segments readable names using simple heuristics
    st.markdown("### Interpretable segment labels (auto-suggested)")
    label_rows = []
    for sid in sorted(seg_out["segment_id"].unique()):
        row = profile.loc[sid]

        # proxies
        spend = row.get("monthly_spend_aed", np.nan)
        value = row.get("how_important_is_value_for_money_in_purchasing_bottled_water", np.nan)
        taste = row.get("how_important_is_taste_in_purchasing_bottled_water", np.nan)
        benefits = row.get(
            "how_important_is_added_benefits_like_alkaline_zero_sodium_added_minerals_in_purchasing_bottled_water", np.nan
        )
        packaging = row.get("how_important_is_packaging_type_in_purchasing_bottled_water", np.nan)
        brand = row.get("how_important_is_brand_name_in_purchasing_bottled_water", np.nan)

        # crude naming rules (transparent, editable)
        if spend >= np.nanpercentile(profile["monthly_spend_aed"].values, 70) and (benefits >= 3.8 or taste >= 3.8):
            name = "Health-conscious premium buyers"
            targeting = "Premium positioning, functional benefits, source/taste proof, higher-end retail + online bundles."
        elif value >= 4.2 and spend <= np.nanpercentile(profile["monthly_spend_aed"].values, 40):
            name = "Price-sensitive mass consumers"
            targeting = "Value packs, aggressive shelf promos, clear AED-per-liter economics, hypermarket focus."
        elif packaging >= 4.0:
            name = "Packaging-first convenience shoppers"
            targeting = "Format innovation (grab-and-go), channel fit, convenience stores + delivery partnerships."
        elif brand >= 4.0:
            name = "Brand-led trust seekers"
            targeting = "Credibility cues, consistent identity, restaurant placement, influencer trust loops."
        else:
            name = "Mainstream practical buyers"
            targeting = "Balanced proposition, wide distribution, simple messaging: availability + taste + value."

        label_rows.append(
            {
                "segment_id": sid,
                "suggested_name": name,
                "share": float(profile.loc[sid, "share"]),
                "targeting_play": targeting,
            }
        )

    labels_df = pd.DataFram_
