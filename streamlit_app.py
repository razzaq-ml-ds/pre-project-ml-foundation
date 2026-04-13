import json
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from project_settings import PROJECT_CONFIG

# Colors
BG_COLOR      = "#323437"
SURFACE_COLOR = "#2c2e31"
TEXT_COLOR    = "#d1d0c5"
MUTED_TEXT    = "#8f908a"
ACCENT_COLOR  = "#e2b714"
RISK_COLOR    = "#ca4754"
GRID_COLOR    = "#45484d"
WARM_SCALE    = ["#4a4d52", "#7a6a32", ACCENT_COLOR]
RISK_SCALE    = ["#4a4d52", "#7a4b51", RISK_COLOR]

st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {BG_COLOR};
            color: {TEXT_COLOR};
        }}
        .block-container {{
            padding-top: 2rem;
            max-width: 1280px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_dataset():
    return pd.read_csv(PROJECT_CONFIG["data"]["path"])

@st.cache_resource
def load_artifacts():
    loaded_model        = joblib.load("models/model.pkl")
    loaded_preprocessor = joblib.load("models/preprocessor.pkl")
    with open("models/experiment_results.json") as f:
        loaded_results = json.load(f)
    return loaded_model, loaded_preprocessor, loaded_results

df                          = load_dataset()
model, preprocessor, exp    = load_artifacts()
target_col                  = PROJECT_CONFIG["data"]["target_column"]

df["Churn Label"] = df[target_col].map({0: "No Churn", 1: "Churn"})

# ── Simple title ──────────────────────────────────────────────────────────────
st.title(PROJECT_CONFIG["project"]["name"])
st.caption("Interactive dashboard for churn behavior and model diagnostics.")

st.header("1. Data Overview")

row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    churn_counts = df["Churn Label"].value_counts().reset_index()
    churn_counts.columns = ["Churn Status", "Count"]
    fig = px.pie(
        churn_counts,
        names="Churn Status",
        values="Count",
        hole=0.5,
        color="Churn Status",
        color_discrete_map={"No Churn": ACCENT_COLOR, "Churn": RISK_COLOR},
        title="Customer Churn Mix",
    )
    st.plotly_chart(fig, use_container_width=True)

with row1_col2:
    geo_churn = (
        df.groupby("Geography")[target_col]
        .mean()
        .reset_index(name="Churn Rate")
        .sort_values("Churn Rate", ascending=False)
    )
    fig = px.bar(
        geo_churn,
        x="Geography",
        y="Churn Rate",
        color="Churn Rate",
        text="Churn Rate",
        color_continuous_scale=RISK_SCALE,
        title="Churn Rate by Geography",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with row1_col3:
    gender_churn = (
        df.groupby("Gender")[target_col]
        .mean()
        .reset_index(name="Churn Rate")
        .sort_values("Churn Rate", ascending=False)
    )
    fig = px.bar(
        gender_churn,
        x="Gender",
        y="Churn Rate",
        color="Gender",
        text="Churn Rate",
        color_discrete_sequence=[ACCENT_COLOR, "#6d92b5"],
        title="Churn Rate by Gender",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    fig = px.histogram(
        df,
        x="Age",
        color="Churn Label",
        nbins=30,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={"No Churn": ACCENT_COLOR, "Churn": RISK_COLOR},
        title="Age Distribution by Churn",
    )
    st.plotly_chart(fig, use_container_width=True)

with row2_col2:
    fig = px.box(
        df,
        x="Churn Label",
        y="Balance",
        color="Churn Label",
        color_discrete_map={"No Churn": ACCENT_COLOR, "Churn": RISK_COLOR},
        title="Balance Spread by Churn",
    )
    st.plotly_chart(fig, use_container_width=True)

with row2_col3:
    product_churn = (
        df.groupby("Num Of Products")[target_col]
        .mean()
        .reset_index(name="Churn Rate")
        .sort_values("Num Of Products")
    )
    product_churn["Num Of Products"] = product_churn["Num Of Products"].astype(str)
    fig = px.bar(
        product_churn,
        x="Num Of Products",
        y="Churn Rate",
        color="Churn Rate",
        text="Churn Rate",
        color_continuous_scale=WARM_SCALE,
        title="Churn Rate by Number of Products",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

