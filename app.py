# ======================================================
# IOTA Water UAE | GTM Analytics Dashboard (UAE Survey)
# End-to-end: KPI metrics + Correlation heatmap + Regression + STP Segmentation
# Streamlit Cloud safe: Loads CSV from local repo path first
# ======================================================

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm


# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="IOTA Water UAE | GTM Dashboard", layout="wide")

st.title("IOTA Water UAE | GTM Analytics Dashboard")
st.caption(
    "Decision-grade GTM insights from UAE bottled water consumer research (n=200). "
    "Includes KPIs, correlation heatmap, regression, and STP segmentation."
)

# --------------------------
# Local data load (repo path)
# --------------------------
DATA_PATH = "data/Research_for_bottled_water_UAE_200_respondents.csv"

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

df_raw = load_data(DATA_PATH)

if df_raw is None:
    st.error(
        f"Dataset not found at `{DATA_PATH}`.\n\n"
        "Fix:\n"
        "1) In GitHub repo, create folder `data/`\n"
        f"2) Upload CSV named exactly: `Research_for_bottled_water_UAE_200_respondents.csv`\n"
        "3) Redeploy Streamlit Cloud"
    )
    st.stop()

# --------------------------
# Helpers
# --------------------------
def clean_col(col: str) -> str:
    col = str(col).lower().strip()
    col = re.sub(r"[^\w\s]", " ", col)
    col = re.sub(r"\s+", "_", col)
    return col.strip("_")

def median_impute(s: pd.Series) -> pd.Series:
    if s.dropna().empty:
        return s
    return s.fillna(s.median())

def mode_impute(s: pd.Series) -> pd.Series:
    if s.dropna().empty:
        return s
    return s.fillna(s.mode().iloc[0])

def safe_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def spend_to_aed(x) -> float:
    """Map categorical spend ranges to AED midpoint proxy."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()

    if "below" in s and "50" in s:
        return 25.0
    if "50" in s and "100" in s:
        return 75.0
    if "100" in s and "200" in s:
        return 150.0
    if "200" in s and "300" in s:
        return 250.0
    if "300" in s and "500" in s:
        return 400.0
    if "500" in s or "above" in s:
        return 600.0

    # fallback parse numbers
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan

def freq_score(x) -> float:
    """Ordinal scoring for purchase frequency."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "more than once a week" in s:
        return 5
    if "once a week" in s:
        return 4
    if "fortnight" in s:
        return 3
    if "once a month" in s or "monthly" in s:
        return 2
    if "rare" in s:
        return 1
    if "never" in s:
        return 0
    return np.nan

def yes_no_to_binary(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["yes", "y", "true", "1"]:
        return 1.0
    if s in ["no", "n", "false", "0"]:
        return 0.0
    return np.nan


# --------------------------
# Clean + Standardize
# --------------------------
df = df_raw.copy()
df.columns = [clean_col(c) for c in df.columns]

# Try to detect key columns (based on your dataset)
COL_SPEND = "what_is_your_average_monthly_spent_on_water_in_a_month"
COL_FREQ  = "how_often_do_you_purchase_packaged_drinking_water"
COL_EATOUT = "how_often_do_you_eat_out"
COL_BUY_EATOUT = "do_you_buy_water_while_eating_out"
COL_CHANNEL = "where_do_you_usually_buy_bottled_water"
COL_PACK = "what_size_of_bottled_water_do_you_buy_most_often"

# Feature engineering
if safe_exists(df, COL_SPEND):
    df["monthly_spend_aed"] = df[COL_SPEND].apply(spend_to_aed)
else:
    df["monthly_spend_aed"] = np.nan

if safe_exists(df, COL_FREQ):
    df["purchase_freq_score"] = df[COL_FREQ].apply(freq_score)
else:
    df["purchase_freq_score"] = np.nan

if safe_exists(df, COL_BUY_EATOUT):
    df["buys_water_when_eating_out"] = df[COL_BUY_EATOUT].apply(yes_no_to_binary)
else:
    df["buys_water_when_eating_out"] = np.nan

# Impute missing values
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

for c in num_cols:
    df[c] = median_impute(df[c])

for c in cat_cols:
    df[c] = mode_impute(df[c])

# ✅ Define predictors globally (THIS FIXES YOUR NameError)
importance_cols = [c for c in df.columns if c.startswith("how_important")]
predictors = importance_cols.copy()

# Add a few behavior predictors if present
for extra in ["purchase_freq_score", "buys_water_when_eating_out"]:
    if extra in df.columns:
        predictors.append(extra)

# Keep only numeric predictors
predictors = [c for c in predictors if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

# --------------------------
# Sidebar navigation
# --------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Data Overview",
            "KPI Metrics",
            "Correlation Heatmap",
            "Regression",
            "Segmentation (STP)",
            "Positioning",
        ],
        index=0
    )

    st.divider()
    st.caption("Data source: local repo CSV")
    st.code(DATA_PATH)


# ======================================================
# 1) DATA OVERVIEW
# ======================================================
if page == "Data Overview":
    st.subheader("Data Overview & Health Check")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{df.shape[0]:,}")
    c2.metric("Variables", f"{df.shape[1]:,}")
    c3.metric("Numeric cols", f"{len(num_cols):,}")
    c4.metric("Drivers detected", f"{len(importance_cols):,}")

    with st.expander("Preview (first 25 rows)", expanded=True):
        st.dataframe(df.head(25), use_container_width=True)

    st.info(
        "So what? If this page loads cleanly, your deployment and data pipeline are stable. "
        "Everything else is just analytics on top."
    )


# ======================================================
# 2) KPI METRICS (Executive tiles)
# ======================================================
elif page == "KPI Metrics":
    st.subheader("KPI Metrics (Executive Snapshot)")

    # KPI 1: Average monthly spend (AED proxy)
    avg_spend = float(df["monthly_spend_aed"].mean()) if "monthly_spend_aed" in df.columns else np.nan
    med_spend = float(df["monthly_spend_aed"].median()) if "monthly_spend_aed" in df.columns else np.nan

    # KPI 2: Purchase frequency score average
    avg_freq = float(df["purchase_freq_score"].mean()) if "purchase_freq_score" in df.columns else np.nan

    # KPI 3: % who buy water while eating out
    pct_eatout = float(df["buys_water_when_eating_out"].mean() * 100) if "buys_water_when_eating_out" in df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Monthly Spend (AED)", f"{avg_spend:,.0f}")
    c2.metric("Median Monthly Spend (AED)", f"{med_spend:,.0f}")
    c3.metric("Avg Purchase Frequency Score", f"{avg_freq:.2f}")
    c4.metric("% Buy Water When Eating Out", f"{pct_eatout:.1f}%")

    st.divider()

    # Channel split
    if safe_exists(df, COL_CHANNEL):
        vc = df[COL_CHANNEL].value_counts().reset_index()
        vc.columns = ["channel", "respondents"]
        fig = px.pie(vc, names="channel", values="respondents", title="Preferred Purchase Channel Split")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "So what? This directly informs distribution priorities (hypermarkets vs online vs convenience). "
            "GTM implication: put your budget where your buyers already shop."
        )
    else:
        st.warning("Purchase channel column not found in dataset.")


# ======================================================
# 3) CORRELATION HEATMAP (numeric drivers + behavior)
# ======================================================
elif page == "Correlation Heatmap":
    st.subheader("Correlation Heatmap (Drivers + Behavior)")

    heat_cols = []
    for c in importance_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            heat_cols.append(c)

    for c in ["monthly_spend_aed", "purchase_freq_score", "buys_water_when_eating_out"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            heat_cols.append(c)

    if len(heat_cols) < 3:
        st.error(
            "Not enough numeric columns for a correlation heatmap. "
            "Check that your 'how_important...' columns are numeric (1–5)."
        )
        st.stop()

    corr = df[heat_cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap"
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "So what? Correlation shows what tends to move together (not causality). "
        "GTM implication: if a driver strongly correlates with spend, it’s a good candidate for positioning and messaging."
    )


# ======================================================
# 4) REGRESSION (WTP proxy = monthly_spend_aed)
# ======================================================
elif page == "Regression":
    st.subheader("Regression: What Drives Willingness-to-Pay (Proxy)")

    if "monthly_spend_aed" not in df.columns or df["monthly_spend_aed"].isna().all():
        st.error("Monthly spend proxy not available. Check your spend column mapping.")
        st.stop()

    if len(predictors) < 3:
        st.error("Not enough predictors for regression. Need at least 3 numeric drivers.")
        st.stop()

    model_df = df[["monthly_spend_aed"] + predictors].replace([np.inf, -np.inf], np.nan).dropna()

    X = sm.add_constant(model_df[predictors])
    y = model_df["monthly_spend_aed"]

    model = sm.OLS(y, X).fit()

    results = pd.DataFrame({
        "feature": model.params.index,
        "coef": model.params.values,
        "p_value": model.pvalues.values
    }).sort_values("coef", ascending=False)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.dataframe(results, use_container_width=True, height=520)

    with c2:
        st.metric("R-squared", f"{model.rsquared:.3f}")
        st.metric("Observations used", f"{len(model_df):,}")

        sig = results[(results["feature"] != "const") & (results["p_value"] < 0.05)].head(6)
        if sig.empty:
            st.warning("No statistically significant drivers at p < 0.05 in this sample.")
        else:
            st.markdown("**Top significant drivers (p < 0.05):**")
            for _, r in sig.iterrows():
                direction = "↑" if r["coef"] > 0 else "↓"
                st.write(f"- {direction} `{r['feature']}` (coef={r['coef']:.2f})")

    st.info(
        "So what? This is your data-backed story: which drivers are associated with higher spend. "
        "GTM implication: align product claims + pricing + channel to the strongest drivers."
    )


# ======================================================
# 5) SEGMENTATION (STP) - KMeans
# ======================================================
elif page == "Segmentation (STP)":
    st.subheader("Market Segmentation (STP)")

    # Build segmentation feature set: drivers + spend + behavior
    seg_features = []
    seg_features += [c for c in importance_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    for c in ["monthly_spend_aed", "purchase_freq_score", "buys_water_when_eating_out"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            seg_features.append(c)

    if len(seg_features) < 4:
        st.error(
            "Not enough features to create stable segments. "
            "Need at least 4 numeric columns (drivers + behavior/spend)."
        )
        st.stop()

    seg_df = df[seg_features].replace([np.inf, -np.inf], np.nan).dropna()

    k = st.slider("Number of segments (K)", min_value=3, max_value=6, value=4)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(seg_df)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    seg_df["segment"] = km.fit_predict(Xs)

    # Segment sizes
    sizes = seg_df["segment"].value_counts().sort_index().reset_index()
    sizes.columns = ["segment", "respondents"]

    c1, c2 = st.columns([0.7, 1.3])
    with c1:
        st.markdown("### Segment sizes")
        st.dataframe(sizes, use_container_width=True, height=250)

    with c2:
        fig = px.bar(sizes, x="segment", y="respondents", title="Segment Size")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Segment profiles
    st.markdown("### Segment profiles (averages)")
    profile = seg_df.groupby("segment")[seg_features].mean().reset_index()
    st.dataframe(profile, use_container_width=True, height=420)

    st.success(
        "GTM implication: pick 1–2 primary segments to win first. "
        "Trying to win every segment on day 1 turns the brand into generic bottled water with extra meetings."
    )


# ======================================================
# 6) POSITIONING (Perceptual-style mapping using proxies)
# ======================================================
elif page == "Positioning":
    st.subheader("Perceptual Positioning (Proxy Map)")

    # Proxies based on your dataset’s driver ratings
    # Price sensitivity proxy: value-for-money importance (if present)
    # Quality proxy: taste + source importance (if present)
    price_col = None
    quality_cols = []

    for c in importance_cols:
        if "value_for_money" in c:
            price_col = c
        if "taste" in c or "source_of_water" in c:
            quality_cols.append(c)

    if price_col is None or len(quality_cols) == 0:
        st.warning(
            "Could not auto-detect the pricing/quality proxy columns. "
            "Your dataset may use different wording in column names."
        )
        st.write("Detected importance columns:")
        st.write(importance_cols)
        st.stop()

    df_map = df.copy()
    df_map["price_proxy"] = df_map[price_col]
    df_map["quality_proxy"] = df_map[quality_cols].mean(axis=1)

    fig = px.scatter(
        df_map,
        x="price_proxy",
        y="quality_proxy",
        title="Perceptual-style map: Price Sensitivity (proxy) vs Quality Seeking (proxy)",
        labels={
            "price_proxy": "Price sensitivity proxy (higher = cares more about value-for-money)",
            "quality_proxy": "Quality-seeking proxy (avg of taste + source importance)"
        }
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "So what? This helps you choose a positioning wedge. "
        "GTM implication: if your target segment is high quality-seeking, win with proof (source/taste/benefits) and premium cues. "
        "If your target segment is high price-sensitive, win with pack economics and distribution dominance."
    )
