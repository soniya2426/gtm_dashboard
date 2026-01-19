# ======================================================
# IOTA Water UAE | GTM Analytics Dashboard
# Streamlit Cloud–safe version (local CSV load)
# ======================================================

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm

# ======================================================
# Page config
# ======================================================
st.set_page_config(
    page_title="IOTA Water UAE | GTM Dashboard",
    layout="wide"
)

st.title("IOTA Water UAE | GTM Analytics Dashboard")
st.caption(
    "Decision-grade Go-To-Market analytics based on UAE consumer research (n=200). "
    "Designed for founders, strategy teams, and investors."
)

# ======================================================
# Load data (LOCAL FIRST – Streamlit Cloud safe)
# ======================================================
DATA_PATH = "data/Research_for_bottled_water_UAE_200_respondents.csv"

@st.cache_data(show_spinner=False)
def load_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

df_raw = load_data(DATA_PATH)

if df_raw is None:
    st.error(
        f"Dataset not found.\n\nExpected file at:\n`{DATA_PATH}`\n\n"
        "Check that the CSV exists in the GitHub repo and was pushed correctly."
    )
    st.stop()

st.success("Dataset loaded successfully")

# ======================================================
# Utilities
# ======================================================
def clean_col(col):
    col = col.lower().strip()
    col = re.sub(r"[^\w\s]", " ", col)
    col = re.sub(r"\s+", "_", col)
    return col.strip("_")

def mode_impute(s):
    return s.fillna(s.mode().iloc[0]) if not s.mode().empty else s

def median_impute(s):
    return s.fillna(s.median())

def spend_to_aed(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if "below" in s: return 25
    if "50" in s and "100" in s: return 75
    if "100" in s and "200" in s: return 150
    if "200" in s and "300" in s: return 250
    if "300" in s: return 400
    if "500" in s or "above" in s: return 600
    return np.nan

def freq_score(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if "more than once" in s: return 5
    if "once a week" in s: return 4
    if "fortnight" in s: return 3
    if "month" in s: return 2
    if "rare" in s: return 1
    return 0

# ======================================================
# Cleaning
# ======================================================
df = df_raw.copy()
df.columns = [clean_col(c) for c in df.columns]

# Numeric spend proxy
if "what_is_your_average_monthly_spent_on_water_in_a_month" in df.columns:
    df["monthly_spend_aed"] = df[
        "what_is_your_average_monthly_spent_on_water_in_a_month"
    ].apply(spend_to_aed)

# Frequency proxy
if "how_often_do_you_purchase_packaged_drinking_water" in df.columns:
    df["purchase_freq_score"] = df[
        "how_often_do_you_purchase_packaged_drinking_water"
    ].apply(freq_score)

# Identify numeric & categorical
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = [c for c in df.columns if c not in num_cols]

for c in num_cols:
    df[c] = median_impute(df[c])

for c in cat_cols:
    df[c] = mode_impute(df[c])

# ======================================================
# Sidebar navigation
# ======================================================
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Data Overview",
            "Consumer Insights",
            "Hypothesis Testing",
            "Segmentation (STP)",
            "Positioning"
        ]
    )

# ======================================================
# DATA OVERVIEW
# ======================================================
if page == "Data Overview":
    st.subheader("Data Overview & Health Check")

    c1, c2, c3 = st.columns(3)
    c1.metric("Respondents", df.shape[0])
    c2.metric("Variables", df.shape[1])
    c3.metric("Numeric Features", len(num_cols))

    st.dataframe(df.head(25), use_container_width=True)

    st.info(
        "So what? This confirms the dataset is complete, clean, and usable for GTM decisions."
    )

# ======================================================
# CONSUMER INSIGHTS
# ======================================================
elif page == "Consumer Insights":
    st.subheader("Consumer Behavior & Drivers")

    if "how_often_do_you_purchase_packaged_drinking_water" in df.columns:
        fig = px.histogram(
            df,
            x="how_often_do_you_purchase_packaged_drinking_water",
            title="Purchase Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)

    importance_cols = [c for c in df.columns if "how_important" in c]

    if importance_cols:
        imp = df[importance_cols].mean().sort_values()
        fig = px.bar(
            imp,
            orientation="h",
            title="Purchase Drivers (Average Importance)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        "GTM insight: Focus messaging and product design on the top 2–3 purchase drivers."
    )

# ======================================================
# HYPOTHESIS TESTING
# ======================================================
elif page == "Hypothesis Testing":
    st.subheader("Regression: Drivers of Willingness to Pay")

    if "monthly_spend_aed" not in df.columns:
        st.error("Monthly spend proxy not available.")
        st.stop()

    predictors = [c for c in df.columns if "how_important" in c]

    model_df = df[["monthly_spend_aed"] + predictors].dropna()

    X = sm.add_constant(model_df[predictors])
    y = model_df["monthly_spend_aed"]

    model = sm.OLS(y, X).fit()

    results = pd.DataFrame({
        "Coefficient": model.params,
        "p_value": model.pvalues
    }).sort_values("Coefficient", ascending=False)

    st.dataframe(results, use_container_width=True)
    st.metric("R-squared", round(model.rsquared, 3))

    st.info(
        "So what? Statistically significant drivers directly inform pricing, claims, and premium positioning."
    )

# ======================================================
# SEGMENTATION
# ======================================================
elif page == "Segmentation (STP)":
    st.subheader("Market Segmentation")

    seg_cols = predictors + ["monthly_spend_aed", "purchase_freq_score"]
    seg_cols = [c for c in seg_cols if c in df.columns]

    seg_df = df[seg_cols].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(seg_df)

    k = st.slider("Number of segments", 3, 6, 4)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    seg_df["segment"] = km.fit_predict(X)

    st.dataframe(
        seg_df.groupby("segment").mean(),
        use_container_width=True
    )

    fig = px.histogram(seg_df, x="segment", title="Segment Size")
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "GTM implication: choose 1–2 priority segments to win first, not all at once."
    )

# ======================================================
# POSITIONING
# ======================================================
elif page == "Positioning":
    st.subheader("Perceptual Positioning")

    if not predictors:
        st.warning("Insufficient drivers for perceptual mapping.")
        st.stop()

    df["quality_proxy"] = df[[c for c in predictors if "taste" in c or "source" in c]].mean(axis=1)
    df["price_proxy"] = df[[c for c in predictors if "value" in c]].mean(axis=1)

    fig = px.scatter(
        df,
        x="price_proxy",
        y="quality_proxy",
        title="Price Sensitivity vs Quality Seeking (Consumer-Level)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Positioning takeaway: decide whether IOTA wins on quality proof or value economics."
    )
