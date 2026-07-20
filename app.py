"""Streamlit application for smart energy prediction and optimization."""

from pathlib import Path
import streamlit as st
from predict import load_model_and_scaler, make_prediction
from optimization import optimize_energy

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Smart_Energy_Model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


st.set_page_config(
    page_title="Smart Energy Prediction",
    page_icon="⚡",
    layout="centered",
)

st.markdown("""
<style>

.stApp{
    background:#f5f7fb;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1400px;
}

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Cards */

.card{
    background:white;
    border-radius:18px;
    padding:25px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.08);
    margin-bottom:25px;
}

.big-card{
    background:linear-gradient(135deg,#eaffef,#f8fff9);
    border:2px solid #ccf0d3;
    border-radius:20px;
    padding:30px;
}

.metric-card{
    background:white;
    border-radius:18px;
    padding:20px;
    text-align:center;
    border:1px solid #eeeeee;
}

.title{
    font-size:42px;
    font-weight:700;
    color:#16254c;
}

.subtitle{
    color:#606b86;
    font-size:18px;
}

.green{
    color:#2ebf66;
    font-weight:bold;
}

.blue-btn button{
    width:100%;
    background:linear-gradient(90deg,#2d6cff,#19c37d);
    color:white;
    height:55px;
    border-radius:12px;
    border:none;
    font-size:18px;
}

div.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#2962ff,#18c37c);
    color:white;
    font-weight:bold;
    border-radius:12px;
    height:55px;
    border:none;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_resources():
    return load_model_and_scaler(MODEL_PATH, SCALER_PATH)

model, scaler = load_resources()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="card">

<div class="title">⚡ AI-Powered Smart Energy Consumption Prediction</div>

<div class="subtitle">
Predict household energy consumption and receive optimization recommendations.
</div>

</div>
""", unsafe_allow_html=True)

left, right = st.columns([1,2])

# -----------------------------
# LEFT PANEL
# -----------------------------
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🏠 Input Details")

    household_id = st.number_input(
        "Household ID",
        value=0,
        step=1
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=25.0
    )

    hour = st.slider(
        "Hour",
        0,
        23,
        12
    )

    day = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
    )

    timestamp = st.text_input(
        "Timestamp",
        "2026-07-19 12:00:00"
    )

    predict = st.button("⚡ Predict Energy Consumption")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# RIGHT PANEL
# -----------------------------
with right:

    if predict:

        input_data = {
            "timestamp": timestamp,
            "household_id": household_id,
            "temperature": temperature,
            "hour": hour,
            "day_of_week": [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ].index(day),
        }

        prediction = make_prediction(
            input_data,
            model,
            scaler
        )

        result = optimize_energy(prediction)

        # -------------------------
        # Prediction Card
        # -------------------------

        st.markdown(f"""
        <div class="big-card">

        <h2 style="color:#2ebf66;">✅ Predicted Energy Consumption</h2>

        <h1 style="font-size:60px;color:#2ebf66;">
        {prediction:.2f} kWh
        </h1>

        </div>

        <br>
        """, unsafe_allow_html=True)

        st.markdown("## 📊 Optimization Results")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Category</h4>
            <h2 style="color:#5b4cff;">
            {result["category"]}
            </h2>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Optimized</h4>
            <h2 style="color:#2b6eff;">
            {result["optimized_consumption_kwh"]:.2f}
            </h2>
            <p>kWh</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Energy Saved</h4>
            <h2 style="color:#ff9800;">
            {result["energy_saved_kwh"]:.2f}
            </h2>
            <p>kWh</p>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Money Saved</h4>
            <h2 style="color:#16a34a;">
            ₹{result["estimated_money_saved"]:.2f}
            </h2>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        st.markdown("""
        <div class="card">
        <h2>💡 Recommendations</h2>
        """, unsafe_allow_html=True)

        for rec in result["recommendations"]:
            st.markdown(f"✅ {rec}")

        st.markdown("</div>", unsafe_allow_html=True)

    else:

        st.info("Enter the details on the left and click **Predict Energy Consumption**.")
