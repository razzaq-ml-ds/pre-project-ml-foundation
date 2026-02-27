import requests
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="🚢",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp { background-color: #1c1c1c; color: #ffd700; }
        [data-testid="stSidebar"] { background-color: #2c2c2c; }
        [data-testid="stMetric"] {
            background: #2c2c2c;
            border: 1px solid #444444;
            border-radius: 12px;
            padding: 16px;
        }
        [data-testid="stMetricValue"] { color: #ffd700 !important; font-size: 2rem !important; }
        [data-testid="stMetricLabel"] { color: #ffd700 !important; }
        p, label, .stMarkdown { color: #ffd700 !important; }
        .stSelectbox > div, .stNumberInput > div { background: #2c2c2c !important; }
        .stButton > button {
            background: #4285f4;
            color: #ffd700;
            font-weight: 700;
            border: none;
            border-radius: 8px;
            width: 100%;
            padding: 12px;
            font-size: 1rem;
        }
        .stButton > button:hover { background: #5f9fff; color: #ffd700; }
        h1 { color: #ffffff !important; }
        h2, h3 { color: #ffd700 !important; font-size: 1.5rem; font-weight: bold; }
        #ml-pipeline-dashboard { color: #ffffff; }
        hr { border-color: #444444; }
    </style>




""", unsafe_allow_html=True)

FLASK_URL = "http://localhost:5000"

st.title("🚢 ML Pipeline Dashboard")
st.markdown("**Titanic Survival Prediction** — Logistic Regression · Production-grade ML Template")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Make a Prediction")
    st.markdown("Fill in passenger details and click predict.")

    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        index=2,
        format_func=lambda x: f"{x}{'st' if x == 1 else 'nd' if x == 2 else 'rd'} Class"
    )

    sex = st.selectbox("Sex", options=["male", "female"])

    a1, a2 = st.columns(2)
    with a1:
        age = st.number_input("Age", min_value=0, max_value=100, value=28)
    with a2:
        fare = st.number_input("Fare (£)", min_value=0.0, value=32.0)

    b1, b2 = st.columns(2)
    with b1:
        sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=8, value=0)
    with b2:
        parch = st.number_input("Parents / Children", min_value=0, max_value=6, value=0)

    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡ Predict Survival"):

        payload = {
            "Pclass": int(pclass),
            "Sex": sex,
            "Age": float(age),
            "SibSp": int(sibsp),
            "Parch": int(parch),
            "Fare": float(fare),
            "Embarked": embarked
        }

        try:
            response = requests.post(f"{FLASK_URL}/predict", json=payload, timeout=5)
            result = response.json()

            if "error" in result:
                st.error(f"API Error: {result['error']}")
            else:
                survived = result["prediction"] == 1
                prob_survived = result["probability_survived"] * 100
                prob_not = result["probability_not_survived"] * 100

                st.markdown("<br>", unsafe_allow_html=True)

                if survived:
                    st.success(f"✅ **Survived** — {prob_survived:.1f}% confidence")
                else:
                    st.error(f"❌ **Did Not Survive** — {prob_not:.1f}% confidence")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_survived,
                    title={"text": "Survival Probability %", "font": {"color": "#94a3b8"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                        "bar": {"color": "#4ade80" if survived else "#f87171"},
                        "bgcolor": "#1e293b",
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 50], "color": "#2d0a0a"},
                            {"range": [50, 100], "color": "#052e16"}
                        ],
                        "threshold": {
                            "line": {"color": "#38bdf8", "width": 3},
                            "thickness": 0.75,
                            "value": 50
                        }
                    },
                    number={"suffix": "%", "font": {"color": "#e2e8f0"}}
                ))

                fig_gauge.update_layout(
                    paper_bgcolor="#0f172a",
                    font={"color": "#e2e8f0"},
                    height=250,
                    margin=dict(t=40, b=0, l=20, r=20)
                )

                st.plotly_chart(fig_gauge, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Flask API. Make sure flask_app.py is running on port 5000.")

with col2:
    st.subheader("📊 Model Performance")

    try:
        metrics_response = requests.get(f"{FLASK_URL}/metrics", timeout=5)
        metrics = metrics_response.json()

        accuracy = metrics["accuracy"]
        cm = metrics["confusion_matrix"]
        report = metrics["classification_report"]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Accuracy", f"{accuracy * 100:.1f}%")
        with m2:
            st.metric("Precision", f"{report['1'] ['precision'] * 100:.1f}%")
        with m3:
            st.metric("Recall", f"{report['1'] ['recall'] * 100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            colorscale=[[0, "#0f172a"], [1, "#38bdf8"]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20, "color": "white"},
            showscale=False
        ))

        fig_cm.update_layout(
            title={"text": "Confusion Matrix", "font": {"color": "#94a3b8", "size": 13}},
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font={"color": "#e2e8f0"},
            height=280,
            margin=dict(t=40, b=20, l=20, r=20),
            xaxis={"tickfont": {"color": "#94a3b8"}},
            yaxis={"tickfont": {"color": "#94a3b8"}}
        )

        st.plotly_chart(fig_cm, use_container_width=True)

        fig_pr = go.Figure()

        fig_pr.add_trace(go.Bar(
            name="Precision",
            x=["Did Not Survive (0)", "Survived (1)"],
            y=[report["0"] ["precision"], report["1"] ["precision"]],
            marker_color="#38bdf8",
            marker_line_width=0
        ))

        fig_pr.add_trace(go.Bar(
            name="Recall",
            x=["Did Not Survive (0)", "Survived (1)"],
            y=[report["0"] ["recall"], report["1"] ["recall"]],
            marker_color="#818cf8",
            marker_line_width=0
        ))

        fig_pr.update_layout(
            title={"text": "Precision vs Recall by Class", "font": {"color": "#94a3b8", "size": 13}},
            barmode="group",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font={"color": "#e2e8f0"},
            height=280,
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis={"range": [0, 1], "gridcolor": "#334155", "tickfont": {"color": "#94a3b8"}},
            xaxis={"tickfont": {"color": "#94a3b8"}},
            legend={"font": {"color": "#94a3b8"}}
        )

        st.plotly_chart(fig_pr, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.warning("Cannot load metrics. Make sure flask_app.py is running.")

st.divider()
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "ML Pipeline Template · Layer 1: ML Engine · Layer 2: Flask API · Layer 3: MySQL (coming soon)"
    "</p>",
    unsafe_allow_html=True
)