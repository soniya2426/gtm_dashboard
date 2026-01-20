import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm


# ======================
# Page config
# ======================
st.set_page_config(page_title="IOTA Water UAE | GTM Dashboard", layout="wide")
st.title("IOTA Water UAE | GTM Analytics Dashboard")
st.caption(
    "Interactive GTM analytics for UAE bottled water survey data. "
    "Includes KPI metrics, correlation heatmap, regression, and configurable STP segmentation + perceptual mapping."
)

# ======================
# Data load (local repo)
# ======================
DATA_PATH = "data/Research_for_bottled_water_UAE_200_respondents.csv"

@st.cache_data(show_spinner=False)
def load_data(path: str):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

df_raw = load_data(DATA_PATH)
if df_raw is None:
    st.error(
        f"Dataset not found at `{DATA_PATH}`.\n\n"
        "Fix: Upload your CSV into `data/` folder in GitHub with the exact filename, then redeploy."
    )
    st.stop()

# ======================
# Helpers
# ======================
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

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

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

def freq_score(x) -> float:
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

def encode_for_modeling(df_in: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, list]:
    """
    Builds a modeling matrix:
    - Numeric columns kept numeric
    - Categorical columns one-hot encoded
    Returns (X_df, feature_names)
    """
    X = df_in[cols].copy()

    # Attempt to convert "rating-like" strings to numeric
    for c in X.columns:
        if X[c].dtype == "object":
            # if it looks mostly numeric, coerce
            coerced = pd.to_numeric(X[c], errors="coerce")
            numeric_ratio = coerced.notna().mean()
            if numeric_ratio > 0.75:
                X[c] = coerced

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Impute
    for c in num_cols:
        X[c] = median_impute(X[c])

    for c in cat_cols:
        X[c] = mode_impute(X[c])

    # One-hot encode categoricals
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    return X, X.columns.tolist()

def top_differentiators(profile_df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    Finds columns with the highest variance across segments (simple differentiator heuristic).
    """
    diffs = profile_df.drop(columns=["segment"], errors="ignore").var().sort_values(ascending=False)
    out = diffs.head(top_n).reset_index()
    out.columns = ["attribute", "between_segment_variance"]
    return out


# ======================
# Clean and feature engineering
# ======================
df = df_raw.copy()
df.columns = [clean_col(c) for c in df.columns]

# Known columns (from your dataset pattern)
COL_SPEND = "what_is_your_average_monthly_spent_on_water_in_a_month"
COL_FREQ = "how_often_do_you_purchase_packaged_drinking_water"
COL_BUY_EATOUT = "do_you_buy_water_while_eating_out"

if COL_SPEND in df.columns:
    df["monthly_spend_aed"] = df[COL_SPEND].apply(spend_to_aed)
else:
    df["monthly_spend_aed"] = np.nan

if COL_FREQ in df.columns:
    df["purchase_freq_score"] = df[COL_FREQ].apply(freq_score)
else:
    df["purchase_freq_score"] = np.nan

if COL_BUY_EATOUT in df.columns:
    df["buys_water_when_eating_out"] = df[COL_BUY_EATOUT].apply(yes_no_to_binary)
else:
    df["buys_water_when_eating_out"] = np.nan

# Global impute pass
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

for c in num_cols:
    df[c] = median_impute(df[c])

for c in cat_cols:
    df[c] = mode_impute(df[c])

# Detect driver columns (importance ratings)
importance_cols = [c for c in df.columns if c.startswith("how_important")]

# ======================
# Sidebar navigation + filters
# ======================
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Data Overview", "KPI Metrics", "Correlation Heatmap", "Regression", "Segmentation (STP)", "Positioning & Perceptual Maps"],
        index=0
    )
    st.divider()
    st.caption("Data file loaded from repo:")
    st.code(DATA_PATH)

# ======================
# PAGES
# ======================
if page == "Data Overview":
    st.subheader("Data Overview & Health Check")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{df.shape[0]:,}")
    c2.metric("Variables", f"{df.shape[1]:,}")
    c3.metric("Numeric cols", f"{len(num_cols):,}")
    c4.metric("Driver (importance) cols", f"{len(importance_cols):,}")

    with st.expander("Preview (first 25 rows)", expanded=True):
        st.dataframe(df.head(25), use_container_width=True)

    st.info("So what? If this loads, your data pipeline and Streamlit deployment are stable.")

elif page == "KPI Metrics":
    st.subheader("KPI Metrics (Executive Snapshot)")

    avg_spend = float(df["monthly_spend_aed"].mean()) if "monthly_spend_aed" in df.columns else np.nan
    med_spend = float(df["monthly_spend_aed"].median()) if "monthly_spend_aed" in df.columns else np.nan
    avg_freq = float(df["purchase_freq_score"].mean()) if "purchase_freq_score" in df.columns else np.nan
    pct_eatout = float(df["buys_water_when_eating_out"].mean() * 100) if "buys_water_when_eating_out" in df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Monthly Spend (AED, proxy)", f"{avg_spend:,.0f}")
    c2.metric("Median Monthly Spend (AED, proxy)", f"{med_spend:,.0f}")
    c3.metric("Avg Purchase Frequency Score", f"{avg_freq:.2f}")
    c4.metric("% Buy Water When Eating Out", f"{pct_eatout:.1f}%")

    st.caption(
        "So what? These four numbers already steer big GTM choices: premium vs value, channel intensity, "
        "and whether out-of-home is a meaningful battleground."
    )

elif page == "Correlation Heatmap":
    st.subheader("Correlation Heatmap (choose what to include)")

    numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    default_heat = [c for c in (importance_cols + ["monthly_spend_aed", "purchase_freq_score"]) if c in numeric_candidates]

    selected = st.multiselect(
        "Select numeric columns for correlation",
        options=numeric_candidates,
        default=default_heat if len(default_heat) >= 3 else numeric_candidates[:10]
    )

    if len(selected) < 3:
        st.warning("Select at least 3 numeric columns to build a meaningful heatmap.")
        st.stop()

    corr = df[selected].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "So what? Correlation helps you spot bundles of preferences. "
        "GTM implication: message stacks should align with what consumers mentally group together."
    )

elif page == "Regression":
    st.subheader("Regression (configurable): Choose outcome + drivers")

    if "monthly_spend_aed" not in df.columns:
        st.error("monthly_spend_aed not available. Check spend mapping.")
        st.stop()

    outcome_options = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    y_col = st.selectbox("Dependent variable (Outcome)", outcome_options, index=outcome_options.index("monthly_spend_aed"))

    feature_pool = df.columns.tolist()
    default_features = [c for c in importance_cols if c in feature_pool]
    for extra in ["purchase_freq_score", "buys_water_when_eating_out"]:
        if extra in feature_pool:
            default_features.append(extra)

    X_cols = st.multiselect(
        "Independent variables (Drivers)",
        options=feature_pool,
        default=default_features
    )

    if len(X_cols) < 3:
        st.warning("Select at least 3 independent variables for regression.")
        st.stop()

    model_df = df[[y_col] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if model_df.empty:
        st.error("No usable rows after dropping missing values. Reduce selected columns.")
        st.stop()

    X_encoded, feature_names = encode_for_modeling(model_df, X_cols)
    X = sm.add_constant(X_encoded)
    y = model_df[y_col]

    model = sm.OLS(y, X).fit()

    results = pd.DataFrame({
        "feature": model.params.index,
        "coef": model.params.values,
        "p_value": model.pvalues.values
    }).sort_values("p_value", ascending=True)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.dataframe(results.head(40), use_container_width=True, height=520)

    with c2:
        st.metric("R-squared", f"{model.rsquared:.3f}")
        st.metric("Observations", f"{len(model_df):,}")

        sig = results[(results["feature"] != "const") & (results["p_value"] < 0.05)].head(8)
        st.markdown("**Top significant drivers (p < 0.05):**")
        if sig.empty:
            st.write("None in this configuration (try different columns).")
        else:
            for _, r in sig.iterrows():
                direction = "↑" if r["coef"] > 0 else "↓"
                st.write(f"- {direction} `{r['feature']}` (coef={r['coef']:.2f})")

    st.caption(
        "So what? This turns survey attributes into a data-backed GTM narrative. "
        "Use significant drivers to choose claims, packaging cues, and pricing architecture."
    )

elif page == "Segmentation (STP)":
    st.subheader("Segmentation (STP): Choose any attributes, then cluster")

    st.markdown(
        "Pick the attributes you want to segment on. This supports **demographics + behavior + attitudes**.\n\n"
        "Behind the scenes:\n"
        "- Numeric columns are standardized\n"
        "- Categorical columns are one-hot encoded\n"
        "- K-Means builds segments\n"
    )

    # Attribute picker: let you truly use "all attributes"
    all_cols = df.columns.tolist()

    # Good defaults (you can change)
    default_seg = []
    default_seg += importance_cols
    for extra in ["monthly_spend_aed", "purchase_freq_score", "buys_water_when_eating_out"]:
        if extra in df.columns:
            default_seg.append(extra)

    seg_cols = st.multiselect(
        "Attributes to use for segmentation",
        options=all_cols,
        default=default_seg
    )

    if len(seg_cols) < 4:
        st.warning("Choose at least 4 attributes to create stable segments.")
        st.stop()

    k = st.slider("Number of segments (K)", 3, 8, 4)

    seg_work = df[seg_cols].replace([np.inf, -np.inf], np.nan).copy()

    # Encode (numeric + one-hot categoricals)
    X_encoded, feat_names = encode_for_modeling(seg_work, seg_cols)

    # Standardize for clustering
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_encoded)

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    segments = km.fit_predict(Xs)

    out = df.copy()
    out["segment"] = segments

    # Segment sizes
    sizes = out["segment"].value_counts().sort_index().reset_index()
    sizes.columns = ["segment", "respondents"]

    c1, c2 = st.columns([0.7, 1.3])
    with c1:
        st.markdown("### Segment sizes")
        st.dataframe(sizes, use_container_width=True, height=260)

    with c2:
        fig = px.bar(sizes, x="segment", y="respondents", title="Segment Size")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Segment profiling:
    # - numeric means
    # - categorical modes (top categories)
    st.markdown("### Segment profile (what makes each segment different?)")

    num_profile_cols = [c for c in seg_cols if pd.api.types.is_numeric_dtype(out[c])]
    cat_profile_cols = [c for c in seg_cols if c not in num_profile_cols]

    profile_blocks = []

    if num_profile_cols:
        num_profile = out.groupby("segment")[num_profile_cols].mean().reset_index()
        st.write("**Numeric averages (within segment):**")
        st.dataframe(num_profile, use_container_width=True, height=380)
        profile_blocks.append(("numeric", num_profile))

    if cat_profile_cols:
        st.write("**Top categorical values (within segment):**")
        cat_summary = []
        for seg_id in sorted(out["segment"].unique()):
            seg_slice = out[out["segment"] == seg_id]
            for c in cat_profile_cols:
                top_val = seg_slice[c].astype(str).value_counts().head(1)
                if not top_val.empty:
                    cat_summary.append({
                        "segment": seg_id,
                        "attribute": c,
                        "top_value": top_val.index[0],
                        "share": float(top_val.iloc[0] / len(seg_slice))
                    })
        cat_summary_df = pd.DataFrame(cat_summary)
        st.dataframe(cat_summary_df, use_container_width=True, height=420)

    # Differentiators (simple variance-based)
    if num_profile_cols:
        diffs = top_differentiators(num_profile, top_n=8)
        st.markdown("### Top differentiating attributes (quick heuristic)")
        st.dataframe(diffs, use_container_width=True, height=300)

    st.success(
        "GTM implication: after finding stable segments, pick 1–2 as primary targets and tailor channel + pricing + messaging."
    )

    # Optional: allow export of segmented dataset
    with st.expander("Download segmented dataset"):
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV with segment labels", csv, file_name="iota_segmented_output.csv", mime="text/csv")

elif page == "Positioning & Perceptual Maps":
    st.subheader("Positioning & Perceptual Mapping (choose axes + optional clustering overlay)")

    st.markdown(
        "A perceptual map is only as good as its axes. Here you can select **any numeric attributes** "
        "as X and Y. You can also overlay the latest clustering from the STP page by re-running clustering here."
    )

    # Pick numeric axes
    numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_candidates) < 2:
        st.error("Need at least 2 numeric columns to build perceptual maps.")
        st.stop()

    default_x = "monthly_spend_aed" if "monthly_spend_aed" in numeric_candidates else numeric_candidates[0]
    default_y = importance_cols[0] if importance_cols and importance_cols[0] in numeric_candidates else numeric_candidates[1]

    x_axis = st.selectbox("X axis (numeric)", numeric_candidates, index=numeric_candidates.index(default_x))
    y_axis = st.selectbox("Y axis (numeric)", numeric_candidates, index=numeric_candidates.index(default_y))

    st.divider()
    st.markdown("### Optional: cluster overlay for the perceptual map")

    do_cluster = st.checkbox("Compute clusters and color the map by segment", value=True)
    k = st.slider("K segments (for overlay)", 3, 8, 4) if do_cluster else None

    color_col = None
    df_plot = df.copy()

    if do_cluster:
        # Let user choose which columns to cluster on (not necessarily only x/y)
        default_cluster_cols = list(set([x_axis, y_axis] + importance_cols[:3] + ["purchase_freq_score"]))
        default_cluster_cols = [c for c in default_cluster_cols if c in df.columns]

        cluster_cols = st.multiselect(
            "Attributes to use for clustering overlay",
            options=df.columns.tolist(),
            default=default_cluster_cols
        )

        if len(cluster_cols) >= 2:
            X_encoded, _ = encode_for_modeling(df_plot[cluster_cols], cluster_cols)
            Xs = StandardScaler().fit_transform(X_encoded)
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            df_plot["segment"] = km.fit_predict(Xs)
            color_col = "segment"
        else:
            st.warning("Pick at least 2 columns for clustering overlay, or turn it off.")
            color_col = None

    # Consumer-level perceptual map
    st.markdown("### Perceptual map (consumer-level)")
    fig = px.scatter(
        df_plot,
        x=x_axis,
        y=y_axis,
        color=color_col,
        title=f"{x_axis} vs {y_axis}",
        opacity=0.75
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "So what? This shows how consumers distribute across your chosen axes. "
        "GTM implication: if your target cluster sits in a distinct zone, you can position IOTA to own that zone."
    )

    # Segment centroid map (if clustering enabled)
    if do_cluster and "segment" in df_plot.columns:
        st.markdown("### Segment centroid map (executive-friendly)")
        centroids = df_plot.groupby("segment")[[x_axis, y_axis]].mean().reset_index()
        centroids["size"] = df_plot["segment"].value_counts().sort_index().values

        fig2 = px.scatter(
            centroids,
            x=x_axis,
            y=y_axis,
            size="size",
            color="segment",
            text="segment",
            title="Segment centroids (size-weighted)"
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

        st.success(
            "Positioning implication: pick the segment centroid you want to win first, then design product cues + channel strategy around it."
        )



