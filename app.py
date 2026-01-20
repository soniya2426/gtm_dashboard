import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm


# ======================
# Config
# ======================
st.set_page_config(page_title="IOTA Water UAE | GTM Dashboard", layout="wide")
st.title("IOTA Water UAE | GTM Analytics Dashboard")
st.caption("KPIs + Perceptual Mapping + Regression + STP Segmentation (KMeans)")

DATA_PATH = "data/gip final data.csv"  # ✅ your new filename


# ======================
# Load
# ======================
@st.cache_data(show_spinner=False)
def load_data(path: str):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

df_raw = load_data(DATA_PATH)
if df_raw is None:
    st.error(
        f"Dataset not found at `{DATA_PATH}`.\n\n"
        "Make sure your repo has:\n"
        "data/gip final data.csv\n\n"
        "Then redeploy Streamlit Cloud."
    )
    st.stop()


# ======================
# Helpers
# ======================
def standardize_col(col: str) -> str:
    col = str(col).replace("\ufeff", "")  # remove BOM
    col = col.strip().lower()
    col = re.sub(r"[^\w\s]", " ", col)
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")

def median_impute(s: pd.Series) -> pd.Series:
    return s.fillna(s.median()) if not s.dropna().empty else s

def mode_impute(s: pd.Series) -> pd.Series:
    return s.fillna(s.dropna().mode().iloc[0]) if not s.dropna().empty else s

def spend_to_aed(x) -> float:
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
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan

def freq_to_score(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "more than once a week" in s:
        return 5
    if "once a week" in s or "once a week" in s or "once a week" in s:
        return 4
    if "fortnight" in s or "fortnite" in s:
        return 3
    if "once a month" in s or "month" in s:
        return 2
    if "rare" in s:
        return 1
    if "never" in s:
        return 0
    return np.nan

def yesno_to_bin(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["yes", "y", "true", "1"]:
        return 1.0
    if s in ["no", "n", "false", "0"]:
        return 0.0
    return np.nan

def split_brands(x) -> list:
    if pd.isna(x):
        return []
    return [b.strip() for b in str(x).split(",") if b.strip()]

def encode_for_clustering(df_in: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = df_in[cols].copy()

    # Try numeric coercion when possible
    for c in X.columns:
        if X[c].dtype == "object":
            coerced = pd.to_numeric(X[c], errors="coerce")
            if coerced.notna().mean() > 0.75:
                X[c] = coerced

    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]

    for c in num:
        X[c] = median_impute(X[c])
    for c in cat:
        X[c] = mode_impute(X[c])

    if cat:
        X = pd.get_dummies(X, columns=cat, drop_first=False)

    return X


# ======================
# Clean + feature engineering
# ======================
df = df_raw.copy()
df.columns = [standardize_col(c) for c in df.columns]

# Drop useless empty index-like column if present (your file has Column1)
for junk in ["column1", ""]:
    if junk in df.columns:
        df = df.drop(columns=[junk], errors="ignore")

# Identify key columns by standardized names
col_spend = "what_is_your_average_monthly_spent_on_water_in_a_month"
col_freq = "how_often_do_you_purchase_packaged_drinking_water"
col_eatout = "how_often_do_you_eat_out"
col_buy_eatout = "do_you_buy_water_while_eating_out"
col_channel = "where_do_you_usually_buy_bottled_water"
col_pack = "size_of_bottled_water"
col_awareness = "which_brands_are_you_aware_of"
col_brand_buy = "which_brands_of_bottled_water_do_you_purchase_most_frequently"

# Attribute ratings (your perception map core)
attribute_cols = [
    "value_for_money",
    "packaging_type",
    "added_benefits",
    "source_of_water",
    "availability",
    "taste",
    "brand_name",
    "attractive_promotions",
]
attribute_cols = [c for c in attribute_cols if c in df.columns]

# engineered numeric columns
df["monthly_spend_aed"] = df[col_spend].apply(spend_to_aed) if col_spend in df.columns else np.nan
df["purchase_freq_score"] = df[col_freq].apply(freq_to_score) if col_freq in df.columns else np.nan
df["eatout_freq_score"] = df[col_eatout].apply(freq_to_score) if col_eatout in df.columns else np.nan
df["buys_water_when_eating_out"] = df[col_buy_eatout].apply(yesno_to_bin) if col_buy_eatout in df.columns else np.nan

# Impute
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]
for c in num_cols:
    df[c] = median_impute(df[c])
for c in cat_cols:
    df[c] = mode_impute(df[c])

# ======================
# Sidebar navigation
# ======================
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Data Overview", "KPI Metrics", "Correlation Heatmap", "Regression", "Segmentation (STP)", "Positioning & Perceptual Map"],
        index=1
    )

    st.divider()
    st.caption("Data file:")
    st.code(DATA_PATH)

# ======================
# Pages
# ======================
if page == "Data Overview":
    st.subheader("Data Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Perception attributes detected", f"{len(attribute_cols):,}")
    st.dataframe(df.head(25), use_container_width=True)

elif page == "KPI Metrics":
    st.subheader("KPI Metrics (Executive Snapshot)")

    avg_spend = float(df["monthly_spend_aed"].mean())
    med_spend = float(df["monthly_spend_aed"].median())
    heavy_spend_cut = float(df["monthly_spend_aed"].quantile(0.75))
    heavy_spend_pct = float((df["monthly_spend_aed"] >= heavy_spend_cut).mean() * 100)

    avg_freq = float(df["purchase_freq_score"].mean())
    heavy_freq_pct = float((df["purchase_freq_score"] >= 4).mean() * 100)

    eatout_buy_pct = float(df["buys_water_when_eating_out"].mean() * 100) if "buys_water_when_eating_out" in df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Monthly Spend (AED, proxy)", f"{avg_spend:,.0f}")
    c2.metric("Median Monthly Spend (AED, proxy)", f"{med_spend:,.0f}")
    c3.metric("Heavy Buyers (Top 25%)", f"{heavy_spend_pct:.1f}%")
    c4.metric("High Frequency Buyers (weekly+)", f"{heavy_freq_pct:.1f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Avg Purchase Frequency Score", f"{avg_freq:.2f}")
    c6.metric("% Buy Water While Eating Out", f"{eatout_buy_pct:.1f}%")
    c7.metric("Attributes used (for perception)", f"{len(attribute_cols):,}")

    st.divider()

    # Top channel and pack size
    if col_channel in df.columns:
        vc = df[col_channel].value_counts(normalize=True).reset_index()
        vc.columns = ["channel", "share"]
        fig = px.bar(vc, x="channel", y="share", title="Channel Share")
        fig.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    if col_pack in df.columns:
        vc2 = df[col_pack].value_counts(normalize=True).reset_index()
        vc2.columns = ["pack_size", "share"]
        fig2 = px.bar(vc2, x="pack_size", y="share", title="Pack Size Share")
        fig2.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    # Top drivers
    if attribute_cols:
        means = df[attribute_cols].mean().sort_values(ascending=False).reset_index()
        means.columns = ["driver", "avg_score"]
        fig3 = px.bar(means, x="avg_score", y="driver", orientation="h", title="Top Purchase Drivers (Avg Rating)")
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

    # Brand awareness and purchase
    colA, colB = st.columns(2)

    if col_awareness in df.columns:
        all_aw = []
        for x in df[col_awareness].tolist():
            all_aw.extend(split_brands(x))
        aw = pd.Series(all_aw).value_counts().head(15).reset_index()
        aw.columns = ["brand", "mentions"]
        figA = px.bar(aw, x="brand", y="mentions", title="Top Brand Awareness (mentions)")
        figA.update_layout(height=420)
        colA.plotly_chart(figA, use_container_width=True)

    if col_brand_buy in df.columns:
        buy = df[col_brand_buy].astype(str).value_counts().head(15).reset_index()
        buy.columns = ["brand", "respondents"]
        figB = px.bar(buy, x="brand", y="respondents", title="Most Frequently Purchased Brand")
        figB.update_layout(height=420)
        colB.plotly_chart(figB, use_container_width=True)

    st.info(
        "So what? These KPIs define your launch reality: spend capacity (WTP proxy), buying intensity, "
        "distribution priorities, and what people actually value."
    )

elif page == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")

    numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    default_heat = [c for c in (attribute_cols + ["monthly_spend_aed", "purchase_freq_score"]) if c in numeric_candidates]

    selected = st.multiselect(
        "Select numeric columns",
        options=numeric_candidates,
        default=default_heat if len(default_heat) >= 3 else numeric_candidates[:10]
    )

    if len(selected) < 3:
        st.warning("Select at least 3 numeric columns.")
        st.stop()

    corr = df[selected].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "So what? Correlation shows which perceptions travel together. "
        "GTM implication: position with a coherent bundle of drivers, not random claims."
    )

elif page == "Regression":
    st.subheader("Regression (Configurable)")

    numeric_outcomes = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if "monthly_spend_aed" not in numeric_outcomes:
        st.error("monthly_spend_aed not available.")
        st.stop()

    y_col = st.selectbox("Dependent variable (Outcome)", numeric_outcomes, index=numeric_outcomes.index("monthly_spend_aed"))

    default_X = attribute_cols + ["purchase_freq_score", "eatout_freq_score", "buys_water_when_eating_out"]
    default_X = [c for c in default_X if c in df.columns]

    X_cols = st.multiselect("Independent variables (Drivers)", df.columns.tolist(), default=default_X)
    if len(X_cols) < 3:
        st.warning("Pick at least 3 predictors.")
        st.stop()

    model_df = df[[y_col] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if model_df.empty:
        st.error("No usable rows after dropping missing values. Reduce columns.")
        st.stop()

    # Encode categoricals if user picked any
    X = model_df[X_cols].copy()
    X = pd.get_dummies(X, drop_first=False)
    X = sm.add_constant(X)
    y = model_df[y_col]

    model = sm.OLS(y, X).fit()

    results = pd.DataFrame({
        "feature": model.params.index,
        "coef": model.params.values,
        "p_value": model.pvalues.values
    }).sort_values("p_value")

    c1, c2 = st.columns([1.2, 0.8])
    c1.dataframe(results.head(50), use_container_width=True, height=520)
    c2.metric("R-squared", f"{model.rsquared:.3f}")
    c2.metric("Observations", f"{len(model_df):,}")

    sig = results[(results["feature"] != "const") & (results["p_value"] < 0.05)].head(8)
    c2.markdown("**Top significant drivers (p < 0.05):**")
    if sig.empty:
        c2.write("None in this configuration.")
    else:
        for _, r in sig.iterrows():
            direction = "↑" if r["coef"] > 0 else "↓"
            c2.write(f"- {direction} `{r['feature']}` (coef={r['coef']:.2f})")

    st.info(
        "So what? Significant coefficients are the data-backed drivers of WTP/spend. "
        "GTM implication: focus product + messaging on what statistically moves value."
    )

elif page == "Segmentation (STP)":
    st.subheader("STP Segmentation (KMeans)")

    st.markdown(
        "Choose the attributes to segment on. Categorical variables will be one-hot encoded automatically."
    )

    default_seg = attribute_cols + ["monthly_spend_aed", "purchase_freq_score", "eatout_freq_score", "buys_water_when_eating_out"]
    default_seg = [c for c in default_seg if c in df.columns]

    seg_cols = st.multiselect("Segmentation attributes", df.columns.tolist(), default=default_seg)
    if len(seg_cols) < 4:
        st.warning("Pick at least 4 attributes for stable segmentation.")
        st.stop()

    k = st.slider("Number of segments (K)", 3, 8, 4)

    X = encode_for_clustering(df, seg_cols)
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=25)
    df_seg = df.copy()
    df_seg["segment"] = km.fit_predict(Xs)

    sizes = df_seg["segment"].value_counts().sort_index().reset_index()
    sizes.columns = ["segment", "respondents"]

    c1, c2 = st.columns([0.7, 1.3])
    c1.dataframe(sizes, use_container_width=True, height=240)
    fig = px.bar(sizes, x="segment", y="respondents", title="Segment Size")
    fig.update_layout(height=320)
    c2.plotly_chart(fig, use_container_width=True)

    # Profile: numeric means for selected columns
    prof_cols = [c for c in seg_cols if pd.api.types.is_numeric_dtype(df_seg[c])]
    if prof_cols:
        profile = df_seg.groupby("segment")[prof_cols].mean().reset_index()
        st.markdown("### Segment Profile (numeric means)")
        st.dataframe(profile, use_container_width=True, height=420)

    st.success(
        "GTM implication: pick 1–2 segments to win first, then expand. "
        "Segments should directly map to channel, pricing, and messaging choices."
    )

    st.session_state["latest_segmented_df"] = df_seg  # used by perceptual map page

elif page == "Positioning & Perceptual Map":
    st.subheader("Perceptual Mapping (Best-in-class for your attributes)")

    if not attribute_cols:
        st.error("No attribute rating columns detected for perceptual mapping.")
        st.stop()

    st.markdown(
        "This page uses **PCA** on your attribute ratings to create a clean, data-driven perceptual map. "
        "You can also overlay **KMeans segments**."
    )

    # Compute PCA on attributes
    attrs = df[attribute_cols].copy()
    scaler = StandardScaler()
    attrs_scaled = scaler.fit_transform(attrs)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(attrs_scaled)

    df_map = df.copy()
    df_map["pc1"] = coords[:, 0]
    df_map["pc2"] = coords[:, 1]

    explained = pca.explained_variance_ratio_
    st.caption(f"PCA variance explained: PC1={explained[0]*100:.1f}% | PC2={explained[1]*100:.1f}%")

    # Optional clustering overlay (reuse from STP if available, else compute quick on attributes)
    overlay = st.checkbox("Overlay KMeans segments on the map", value=True)
    if overlay:
        k = st.slider("Segments (for overlay)", 3, 8, 4)

        km = KMeans(n_clusters=k, random_state=42, n_init=25)
        df_map["segment"] = km.fit_predict(attrs_scaled)
        color_col = "segment"
    else:
        color_col = None

    fig = px.scatter(
        df_map,
        x="pc1",
        y="pc2",
        color=color_col,
        opacity=0.75,
        title="Perceptual Map (PCA on attribute ratings)"
    )
    fig.update_layout(height=650, xaxis_title="Perceptual Axis 1 (PC1)", yaxis_title="Perceptual Axis 2 (PC2)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "So what? This map shows how consumer preferences cluster across all your key attributes, not just two chosen axes. "
        "GTM implication: pick a target cluster and position IOTA to dominate that cluster’s driver bundle."
    )

    # Brand-level map: average attribute positions by most purchased brand (if present)
    if col_brand_buy in df.columns:
        st.markdown("### Brand-level perceptual map (averages by most purchased brand)")
        brand_df = df_map.copy()
        brand_df["brand_key"] = brand_df[col_brand_buy].astype(str).str.strip()

        counts = brand_df["brand_key"].value_counts()
        keep = counts[counts >= 5].index.tolist()
        brand_df = brand_df[brand_df["brand_key"].isin(keep)].copy()

        if not brand_df.empty:
            brand_centroids = brand_df.groupby("brand_key")[["pc1", "pc2"]].mean().reset_index()
            brand_centroids["n"] = brand_df["brand_key"].value_counts().values

            fig2 = px.scatter(
                brand_centroids,
                x="pc1",
                y="pc2",
                size="n",
                text="brand_key",
                title="Brand centroid map (only brands with >=5 respondents)"
            )
            fig2.update_traces(textposition="top center")
            fig2.update_layout(height=650)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Not enough repeated brand selections to build a stable brand-level map (need >=5 per brand).")

