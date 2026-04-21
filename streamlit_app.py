import json
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from project_settings import PROJECT_CONFIG

BG_COLOR      = "#000000"
SURFACE_COLOR = "#2c2e31"
CHART_BG      = "#0d0d0d"        
TEXT_COLOR    = "#d1d0c5"
MUTED_TEXT    = "#8f908a"
ACCENT_COLOR  = "#e2b714"
RISK_COLOR    = "#ca4754"
SAFE_COLOR    = "#43a047"
GRID_COLOR    = "#45484d"
WARM_SCALE    = ["#4a4d52", "#7a6a32", ACCENT_COLOR]
RISK_SCALE    = ["#4a4d52", "#7a4b51", RISK_COLOR]

st.set_page_config(
    page_title=PROJECT_CONFIG["project"]["name"],
    layout="wide",
)


st.markdown(
    f"""
    <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(226, 183, 20, 0.08), transparent 24%),
                radial-gradient(circle at top right, rgba(202, 71, 84, 0.07), transparent 22%),
                linear-gradient(180deg, {BG_COLOR} 0%, #25272b 100%);
            color: {TEXT_COLOR};
        }}
        .block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1280px; }}
        .hero-title {{ color: {ACCENT_COLOR}; font-size: 3rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 0.15rem; }}
        .hero-subtitle {{ color: {MUTED_TEXT}; font-size: 1.05rem; max-width: 860px; margin-bottom: 1.4rem; }}
        .section-title {{ color: {TEXT_COLOR}; font-size: 1.7rem; font-weight: 700; margin-top: 1.25rem; margin-bottom: 0.35rem; }}
        .section-copy {{ color: {MUTED_TEXT}; max-width: 900px; margin-bottom: 1rem; }}
        .result-card {{
            background: linear-gradient(180deg, #2f3136 0%, #26282c 100%);
            border: 1px solid #3c3f45; border-radius: 22px;
            padding: 1.2rem 1.3rem; color: {TEXT_COLOR};
            box-shadow: 0 16px 40px rgba(0,0,0,0.18);
        }}
        .result-title {{ font-size: 0.95rem; letter-spacing: 0.04em; text-transform: uppercase; color: {MUTED_TEXT}; }}
        .result-value {{ font-size: 2rem; font-weight: 700; margin: 0.2rem 0 0.4rem 0; }}
        .result-note {{ color: #b8b7af; font-size: 0.95rem; }}
        .small-note {{ color: {MUTED_TEXT}; font-size: 0.9rem; }}
        div[data-testid="stPlotlyChart"] {{
            background: rgba(44,46,49,0.94); border: 1px solid #3c3f45;
            border-radius: 20px; padding: 0.35rem 0.35rem 0 0.35rem;
            box-shadow: 0 10px 28px rgba(0,0,0,0.15);
        }}
        div[data-testid="stDataFrame"] {{
            background: rgba(44,46,49,0.94); border: 1px solid #3c3f45;
            border-radius: 18px; padding: 0.4rem;
        }}
        div[data-testid="stSelectbox"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSlider"] label {{ color: #b8b7af !important; font-weight: 600; }}
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stNumberInput input {{
            background: {SURFACE_COLOR} !important;
            border-color: {GRID_COLOR} !important;
            color: {TEXT_COLOR} !important;
        }}
        .stButton button {{
            background: {ACCENT_COLOR} !important; color: #232427 !important;
            border: none !important; border-radius: 14px !important;
            font-weight: 700 !important; padding: 0.75rem 1rem !important;
            box-shadow: 0 8px 24px rgba(226,183,20,0.25);
        }}
        .stButton button:hover {{ background: #f0c93d !important; }}
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

df                              = load_dataset()
model, preprocessor, exp        = load_artifacts()
target_col                      = PROJECT_CONFIG["data"]["target_column"]

results_df                      = pd.DataFrame(exp["results"])
results_df["model_display"]     = results_df["model_name"].str.replace("_", " ").str.title()
best_result                     = exp["best_results"]
best_model_name                 = exp["best_model_name"]
best_model_display              = best_model_name.replace("_", " ").title()
best_threshold                  = best_result["selected_threshold"]

df["Churn Label"] = df[target_col].map({0: "No Churn", 1: "Churn"})


st.markdown(f"<div class='hero-title'>{PROJECT_CONFIG['project']['name']}</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Interactive dashboard for churn behavior, live customer scoring, and final model diagnostics.</div>", unsafe_allow_html=True)


st.markdown("<div class='section-title'>1. Data Overview</div>", unsafe_allow_html=True)
st.markdown("<div class='section-copy'>First-look business view: how churn is distributed, where it concentrates, and which customer segments deserve attention.</div>", unsafe_allow_html=True)

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
    fig.update_layout(
        height=360, margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR),
        legend=dict(font=dict(color=TEXT_COLOR)),
    )
    st.plotly_chart(fig, use_container_width=True)

with row1_col2:
    geo_churn = (
        df.groupby("Geography")[target_col].mean()
        .reset_index(name="Churn Rate").sort_values("Churn Rate", ascending=False)
    )
    fig = px.bar(
        geo_churn, x="Geography", y="Churn Rate", color="Churn Rate",
        text="Churn Rate", color_continuous_scale=RISK_SCALE, title="Churn Rate by Geography",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(
        height=360, margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE_COLOR,
        font=dict(color=TEXT_COLOR), coloraxis_showscale=False,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=TEXT_COLOR))
    fig.update_yaxes(tickformat=".0%", gridcolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT))
    st.plotly_chart(fig, use_container_width=True)

with row1_col3:
    gender_churn = (
        df.groupby("Gender")[target_col].mean()
        .reset_index(name="Churn Rate").sort_values("Churn Rate", ascending=False)
    )
    fig = px.bar(
        gender_churn, x="Gender", y="Churn Rate", color="Gender",
        text="Churn Rate", color_discrete_sequence=[ACCENT_COLOR, "#6d92b5"],
        title="Churn Rate by Gender",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(
        height=360, margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE_COLOR,
        font=dict(color=TEXT_COLOR), showlegend=False,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=TEXT_COLOR))
    fig.update_yaxes(tickformat=".0%", gridcolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT))
    st.plotly_chart(fig, use_container_width=True)

row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    fig = px.histogram(
        df, x="Age", color="Churn Label", nbins=30, barmode="overlay", opacity=0.75,
        color_discrete_map={"No Churn": ACCENT_COLOR, "Churn": RISK_COLOR},
        title="Age Distribution by Churn",
    )
    fig.update_layout(
        height=360, margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE_COLOR,
        font=dict(color=TEXT_COLOR), legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=TEXT_COLOR))
    fig.update_yaxes(gridcolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT))
    st.plotly_chart(fig, use_container_width=True)

with row2_col2:
    fig = px.box(
        df, x="Churn Label", y="Balance", color="Churn Label",
        color_discrete_map={"No Churn": ACCENT_COLOR, "Churn": RISK_COLOR},
        title="Balance Spread by Churn",
    )
    fig.update_layout(
        height=360, margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE_COLOR,
        font=dict(color=TEXT_COLOR), showlegend=False, legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=TEXT_COLOR))
    fig.update_yaxes(gridcolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT))
    st.plotly_chart(fig, use_container_width=True)

with row2_col3:
    product_churn = (
        df.groupby("Num Of Products")[target_col].mean()
        .reset_index(name="Churn Rate").sort_values("Num Of Products")
    )
    product_churn["Num Of Products"] = product_churn["Num Of Products"].astype(str)
    fig = px.bar(
        product_churn, x="Num Of Products", y="Churn Rate", color="Churn Rate",
        text="Churn Rate", color_continuous_scale=WARM_SCALE,
        title="Churn Rate by Number of Products",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(
        height=360, margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE_COLOR,
        font=dict(color=TEXT_COLOR), coloraxis_showscale=False, legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=TEXT_COLOR))
    fig.update_yaxes(tickformat=".0%", gridcolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='section-title'>2. Prediction Lab</div>", unsafe_allow_html=True)
st.markdown("<div class='section-copy'>Enter one customer profile, score it with the deployed model, and inspect the churn probability against the saved operating threshold.</div>", unsafe_allow_html=True)

lab_left, lab_right = st.columns([1.35, 1])

with lab_left:
    st.markdown("#### Customer Inputs")
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        credit_score    = st.slider("Credit Score", 300, 900, 650)
        geography       = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender          = st.selectbox("Gender", ["Female", "Male"])
        age             = st.slider("Age", 18, 92, 38)
        tenure          = st.slider("Tenure", 0, 10, 5)

    with input_col2:
        balance          = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)
        num_products     = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        has_credit_card  = st.selectbox("Has Credit Card", [0, 1], index=1)
        is_active_member = st.selectbox("Is Active Member", [0, 1], index=1)
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0)

    predict_button = st.button("Predict Churn", type="primary", use_container_width=True)

with lab_right:
    st.markdown("#### Live Model Setup")
    model_options    = results_df["model_name"].tolist()
    best_model_index = model_options.index(best_model_name)

    st.caption("Live prediction uses the saved best model artifact from training.")
    st.markdown(
        f"""
        <div class='result-card'>
            <div class='result-title'>Current Deployed Model</div>
            <div class='result-value'>{best_model_display}</div>
            <div class='result-note'>Threshold: {best_threshold:.2f}</div>
            <div class='result-note'>F1 Score: {best_result['f1_score']:.3f} | Recall: {best_result['recall']:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if predict_button:
    input_row = pd.DataFrame([{
        "CustomerId": 0, "Surname": "Demo",
        "CreditScore": credit_score, "Geography": geography, "Gender": gender,
        "Age": age, "Tenure": tenure, "Balance": balance,
        "Num Of Products": num_products, "Has Credit Card": has_credit_card,
        "Is Active Member": is_active_member, "Estimated Salary": estimated_salary,
    }])
    transformed_input = preprocessor.transform(input_row)
    churn_probability = float(model.predict_proba(transformed_input)[0][1])
    churn_prediction  = int(churn_probability >= best_threshold)
    prediction_label  = "Will Churn" if churn_prediction == 1 else "Will Not Churn"
    prediction_color  = RISK_COLOR if churn_prediction == 1 else SAFE_COLOR

    result_col1, result_col2 = st.columns([1, 1.05])

    with result_col1:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_probability * 100,
            number={"suffix": "%", "font": {"size": 34, "color": TEXT_COLOR}},
            title={"text": "Churn Risk", "font": {"size": 22, "color": TEXT_COLOR}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED_TEXT},
                "bar":  {"color": ACCENT_COLOR},
                "steps": [
                    {"range": [0, best_threshold * 100],   "color": "#4a4d52"},
                    {"range": [best_threshold * 100, 100], "color": "#6f3d46"},
                ],
                "threshold": {
                    "line": {"color": RISK_COLOR, "width": 4},
                    "thickness": 0.75, "value": best_threshold * 100,
                },
            },
        ))
        gauge_fig.update_layout(
            height=320, margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR),
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with result_col2:
        st.markdown(
            f"""
            <div class='result-card'>
                <div class='result-title'>Prediction Outcome</div>
                <div class='result-value' style='color:{prediction_color};'>{prediction_label}</div>
                <div class='result-note'>Predicted churn probability: {churn_probability:.2%}</div>
                <div class='result-note'>Decision threshold used: {best_threshold:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if churn_prediction == 1:
            st.warning("This customer is above the saved churn-risk threshold.")
        else:
            st.success("This customer is below the saved churn-risk threshold.")
