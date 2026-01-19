import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Interactive GTM Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def clean_bottled_water_df(df: pd.DataFrame) -> pd.DataFrame:
    # drop unnamed cols
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    return df

def get_importance_cols(df: pd.DataFrame) -> list[str]:
    # Your dataset has multiple "How important is ..." Likert columns
    return [c for c in df.columns if c.lower().startswith("how important is")]

def safe_onehot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df[cols].astype(str), drop_first=False)

# -----------------------------
# Sidebar: data loading
# -----------------------------
st.sidebar.title("Data")
use_upload = st.sidebar.toggle("Upload files instead of /data folder", value=False)

bottled_default_path = "data/Research_for_bottled_water_UAE_200.xlsx"
estee_default_path = "data/estee_lauder-dataset_50.xlsx"

bottled_df = None
estee_df = None

if use_upload:
    bottled_file = st.sidebar.file_uploader("Upload bottled water dataset (.xlsx)", type=["xlsx"])
    estee_file = st.sidebar.file_uploader("Upload Estee Lauder dataset (.xlsx)", type=["xlsx"])

    if bottled_file:
        bottled_df = pd.read_excel(bottled_file)
    if estee_file:
        estee_df = pd.read_excel(estee_file)
else:
    if os.path.exists(bottled_default_path):
        bottled_df = load_excel(bottled_default_path)
    if os.path.exists(estee_default_path):
        estee_df = load_excel(estee_default_path)

st.title("Interactive Dashboard (GTM + Research Analytics)")

tabs = st.tabs(["Overview", "Bottled Water: EDA", "Heatmap + Regression", "STP Segmentation", "Perception Map", "Estee Lauder (placeholder)"])

# -----------------------------
# Tab: Overview
# -----------------------------
with tabs[0]:
    st.subheader("Data status")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Bottled water dataset**")
        if bottled_df is None:
            st.error("Not loaded yet. Put the file in /data or upload it from the sidebar.")
        else:
            st.success(f"Loaded: {bottled_df.shape[0]} rows, {bottled_df.shape[1]} columns")
    with c2:
        st.write("**Estee Lauder dataset**")
        if estee_df is None:
            st.warning("Not loaded (OK for now). Add it to /data or upload it.")
        else:
            st.success(f"Loaded: {estee_df.shape[0]} rows, {estee_df.shape[1]} columns")

# -----------------------------
# Bottled Water EDA
# -----------------------------
with tabs[1]:
    if bottled_df is None:
        st.stop()

    df = clean_bottled_water_df(bottled_df)

    st.subheader("Quick preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Key columns (based on your file)
    age_col = "Please specify your Age"
    work_col = "What is your current work status?"
    nat_col = "What is your Nationality?"
    freq_col = "How often do you purchase packaged drinking water?"
    behavior_col = "What best describes your purchase behaviour?"
    size_col = "What size of bottled water do you buy most often?"
    channel_col = "Where do you usually buy bottled water?"
    spend_col = "What is your average monthly spent on water in a Month?"

    # Some files have trailing spaces; try fuzzy fallback
    def find_col(name):
        for c in df.columns:
            if c.strip().lower() == name.strip().lower():
                return c
        return None

    age_col = find_col(age_col) or age_col
    work_col = find_col(work_col) or work_col
    nat_col = find_col(nat_col) or nat_col
    freq_col = find_col(freq_col) or freq_col
    behavior_col = find_col(behavior_col) or behavior_col
    size_col = find_col(size_col) or size_col
    channel_col = find_col(channel_col) or channel_col
    spend_col = find_col(spend_col) or spend_col

    left, right = st.columns(2)

    with left:
        st.write("### Demographics")
        if age_col in df:
            st.plotly_chart(px.histogram(df, x=age_col), use_container_width=True)
        if work_col in df:
            st.plotly_chart(px.histogram(df, x=work_col), use_container_width=True)

    with right:
        if nat_col in df:
            st.write("### Nationality")
            top_nat = df[nat_col].value_counts().head(15).reset_index()
            top_nat.columns = ["Nationality", "Count"]
            st.plotly_chart(px.bar(top_nat, x="Nationality", y="Count"), use_container_width=True)

    st.write("### Purchase behavior")
    c1, c2, c3 = st.columns(3)
    with c1:
        if freq_col in df:
            st.plotly_chart(px.histogram(df, x=freq_col), use_container_width=True)
    with c2:
        if size_col in df:
            st.plotly_chart(px.histogram(df, x=size_col), use_container_width=True)
    with c3:
        if channel_col in df:
            st.plotly_chart(px.histogram(df, x=channel_col), use_container_width=True)

    if spend_col in df:
        st.write("### Monthly spend")
        st.plotly_chart(px.histogram(df, x=spend_col), use_container_width=True)

# -----------------------------
# Heatmap + Regression
# -----------------------------
with tabs[2]:
    if bottled_df is None:
        st.stop()

    df = clean_bottled_water_df(bottled_df)
    imp_cols = get_importance_cols(df)

    st.subheader("Importance drivers")
    if not imp_cols:
        st.error("No importance columns detected.")
        st.stop()

    means = df[imp_cols].mean(numeric_only=True).sort_values(ascending=False).reset_index()
    means.columns = ["Driver", "Mean (1-5)"]
    st.dataframe(means, use_container_width=True)

    st.subheader("Correlation heatmap (importance drivers)")
    corr = df[imp_cols].corr()

    # Plotly heatmap
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Linear regression (example)")
    st.caption("Predict a chosen driver using the other drivers (plus simple controls if you want later).")

    y_col = st.selectbox("Choose dependent variable (Y)", imp_cols, index=0)
    X_cols = [c for c in imp_cols if c != y_col]

    X = df[X_cols].astype(float)
    y = df[y_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)

    st.write(f"**R² on test set:** {r2:.3f}")

    coef = pd.DataFrame({"Feature": X_cols, "Coefficient": model.coef_}).sort_values("Coefficient", ascending=False)
    st.dataframe(coef, use_container_width=True)

# -----------------------------
# STP Segmentation
# -----------------------------
with tabs[3]:
    if bottled_df is None:
        st.stop()

    df = clean_bottled_water_df(bottled_df)
    imp_cols = get_importance_cols(df)

    st.subheader("Segmentation (STP) using clustering")
    st.caption("This creates segments based on importance drivers, then profiles them using behavior + brand choices.")

    k = st.slider("Number of segments (k)", 2, 6, 3)

    X = df[imp_cols].astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    seg = km.fit_predict(Xs)
    df_seg = df.copy()
    df_seg["Segment"] = seg

    st.write("### Segment sizes")
    seg_sizes = df_seg["Segment"].value_counts().sort_index().reset_index()
    seg_sizes.columns = ["Segment", "Count"]
    st.plotly_chart(px.bar(seg_sizes, x="Segment", y="Count"), use_container_width=True)

    st.write("### Segment driver profile (mean importance)")
    profile = df_seg.groupby("Segment")[imp_cols].mean().reset_index()
    st.dataframe(profile, use_container_width=True)

    st.write("### Segment interpretation helper")
    st.caption("Look for the highest drivers per segment. That’s the basis for naming them (e.g., Value Seekers, Brand Loyalists, Health/Benefits).")

# -----------------------------
# Perception Map
# -----------------------------
with tabs[4]:
    if bottled_df is None:
        st.stop()

    df = clean_bottled_water_df(bottled_df)
    imp_cols = get_importance_cols(df)

    st.subheader("Perception map (PCA on importance drivers)")
    st.caption("Each point is a respondent. Color by segment if available.")

    # Build segments again (keep consistent with k=3 default)
    X = df[imp_cols].astype(float)
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    seg = km.fit_predict(Xs)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Segment": seg
    })

    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Segment", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.write("### What the axes mean")
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1_loading", "PC2_loading"])
    loadings["Driver"] = imp_cols
    st.dataframe(loadings.sort_values("PC1_loading", ascending=False), use_container_width=True)

# -----------------------------
# Estee Lauder placeholder
# -----------------------------
with tabs[5]:
    st.subheader("Estee Lauder dataset")
    if estee_df is None:
        st.info("Upload the file or add it to /data. Once loaded, we can mirror the same dashboard logic.")
    else:
        st.dataframe(estee_df.head(30), use_container_width=True)
        st.write("Next: define the matching KPIs, drivers, segmentation variables for this dataset.")

