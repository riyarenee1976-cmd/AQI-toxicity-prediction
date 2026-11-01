
import streamlit as st
import pandas as pd
import json
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AirGuard | Air Quality Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #071b16, #0c2921);
    color: #ecfdf5;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

h1, h2, h3 {
    color: #ecfdf5 !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #cbd5d1;
}

[data-testid="stMetric"] {
    background: #12382c;
    border: 1px solid #285542;
    padding: 20px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #6ee7a5;
}

[data-testid="stMetricLabel"] {
    color: #b6d9c5;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #285542;
}

.stTabs [data-baseweb="tab"] {
    background: #12382c;
    color: #d1fae5;
    border-radius: 10px 10px 0 0;
    padding: 12px 22px;
}

.stTabs [aria-selected="true"] {
    background: #176b48 !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(90deg, #16a36a, #25c47a);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #128653;
    color: white;
    border: none;
}

.stSelectbox, .stNumberInput, .stSlider {
    background: transparent;
}

div[data-baseweb="select"] > div {
    background: #12382c;
    border-color: #285542;
}

input {
    background-color: #12382c !important;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.ag-header {
    background: linear-gradient(110deg, #123d2d, #176b48);
    border: 1px solid #347c57;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.ag-card {
    background: #12382c;
    border: 1px solid #285542;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 15px;
}

.ag-result {
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}

.ag-footer {
    text-align: center;
    color: #86a99a;
    padding: 25px;
    border-top: 1px solid #285542;
    margin-top: 35px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL AND DATA ----------------

@st.cache_resource
def load_model():
    with open("model_meta.json", "r") as f:
        metadata = json.load(f)

    return joblib.load("aqi_model.pkl"), metadata


@st.cache_data
def load_data():
    return pd.read_csv("dashboard_data.csv")


model, meta = load_model()
data = load_data()


# ---------------- CLASSIFICATION STYLE ----------------

STYLE = {
    "Safe": (
        "#16a36a",
        "🟢",
        "Air quality is satisfactory. Normal outdoor activity is fine."
    ),
    "Moderate": (
        "#e5a329",
        "🟡",
        "Air quality is moderate. Sensitive individuals should limit prolonged outdoor activity."
    ),
    "Toxic / Unhealthy": (
        "#e05252",
        "🔴",
        "Air quality is unhealthy. Reduce outdoor exposure and follow local health guidance."
    )
}


# ---------------- HEADER ----------------

st.markdown("""
<div class="ag-header">
    <h1>🌿 AIRGUARD</h1>
    <h3>Intelligent Air Quality Monitoring System</h3>
    <p>
        Machine learning powered air quality classification
        and environmental analytics.
    </p>
</div>
""", unsafe_allow_html=True)


# MODEL INFORMATION

st.markdown("### 🤖 Machine Learning Model")

m1, m2, m3 = st.columns(3)

m1.metric("Algorithm", meta["best_model"])
m2.metric("Test Accuracy", f'{meta["accuracy"] * 100:.2f}%')
m3.metric("Macro F1 Score", f'{meta["macro_f1"]:.3f}')

st.write("")


# ---------------- NAVIGATION ----------------

tab1, tab2, tab3 = st.tabs([
    "🔍  AIR QUALITY PREDICTION",
    "📊  ANALYTICS DASHBOARD",
    "🌱  ABOUT AIRGUARD"
])


# ==================================================
# TAB 1: PREDICTION
# ==================================================

with tab1:

    st.markdown("## 🔍 Air Quality Prediction")
    st.write(
        "Enter pollutant measurements and weather conditions "
        "to estimate the air quality category."
    )

    st.markdown("### 🧪 Pollutant Measurements")

    c1, c2, c3 = st.columns(3)

    with c1:
        pollutant = st.selectbox(
            "Select Pollutant",
            meta["pollutants"]
        )

        p_min = st.number_input(
            "Minimum Concentration",
            min_value=0.0,
            max_value=300.0,
            value=30.0
        )

    with c2:
        p_avg = st.number_input(
            "Average Concentration",
            min_value=0.0,
            max_value=300.0,
            value=70.0
        )

        p_max = st.number_input(
            "Maximum Concentration",
            min_value=0.0,
            max_value=500.0,
            value=120.0
        )

    with c3:
        season = st.selectbox(
            "Season",
            meta["seasons"]
        )

        hour = st.slider(
            "Hour of Day",
            0, 23, 12
        )

    st.markdown("### 🌤️ Weather Conditions")

    w1, w2, w3 = st.columns(3)

    with w1:
        temp = st.slider(
            "Temperature (°C)",
            15.0, 40.0, 28.0
        )

    with w2:
        hum = st.slider(
            "Humidity (%)",
            30.0, 95.0, 60.0
        )

    with w3:
        wind = st.slider(
            "Wind Speed (km/h)",
            0.0, 30.0, 15.0
        )

    st.write("")

    if st.button("🌿 ANALYSE AIR QUALITY"):

        if not (p_min <= p_avg <= p_max):
            st.error(
                "Please ensure Minimum ≤ Average ≤ Maximum concentration."
            )

        else:

            row = pd.DataFrame([{
                "pollutant_min": p_min,
                "pollutant_max": p_max,
                "pollutant_avg": p_avg,
                "Temperature_C": temp,
                "Humidity_%": hum,
                "Wind_Speed_kmh": wind,
                "Hour": hour,
                "pollutant_id": pollutant,
                "Season": season
            }])

            pred = model.predict(row)[0]
            proba = model.predict_proba(row)[0]

            color, icon, advice = STYLE[pred]

            st.markdown(
                f"""
                <div class="ag-result"
                     style="background:{color}20;
                            border:1px solid {color};">
                    <h2 style="color:{color}!important;">
                        {icon} AIR QUALITY RESULT
                    </h2>
                    <h1 style="color:{color}!important;
                               font-size:38px;">
                        {pred}
                    </h1>
                    <p style="color:#ecfdf5;">
                        Your predicted air quality category
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 💡 Health Recommendation")

            st.info(advice)

            st.markdown("### 📈 Prediction Confidence")

            confidence = pd.DataFrame({
                "Confidence (%)": proba * 100
            }, index=model.classes_)

            st.bar_chart(confidence)

            st.caption(
                "Prediction confidence represents the model's "
                "estimated class probabilities, not a guaranteed outcome."
            )


# ==================================================
# TAB 2: ANALYTICS DASHBOARD
# ==================================================

with tab2:

    st.markdown("## 📊 Air Quality Analytics")

    st.write(
        "Explore air quality patterns and pollutant distributions "
        "in the available dataset."
    )

    # DATASET METRICS

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Total Records",
        f"{len(data):,}"
    )

    m2.metric(
        "Cities Covered",
        f"{data['city'].nunique():,}"
    )

    m3.metric(
        "Average AQI",
        f"{data['AQI'].mean():.1f}"
    )

    toxic_percentage = (
        data["Air_Quality_Class"] == "Toxic / Unhealthy"
    ).mean() * 100

    m4.metric(
        "Toxic Readings",
        f"{toxic_percentage:.1f}%"
    )

    st.write("")

    # CHARTS

    left, right = st.columns(2)

    with left:

        st.markdown("### 📌 Air Quality Distribution")

        class_counts = (
            data["Air_Quality_Class"]
            .value_counts()
        )

        st.bar_chart(class_counts)

    with right:

        st.markdown("### 🧪 Pollutant Analysis")

        pollutant_avg = (
            data.groupby("pollutant_id")["pollutant_avg"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(pollutant_avg)

    st.divider()

    # STATE FILTER

    st.markdown("### 🗺️ Explore by State")

    city = st.selectbox(
        "Choose a State",
        ["All"] + sorted(data["state"].dropna().unique())
    )

    view = (
        data if city == "All"
        else data[data["state"] == city]
    )

    st.markdown("### 🏙️ Most Polluted Cities")

    top_cities = (
        view.groupby("city")["AQI"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_cities)

    st.markdown("### 📍 Air Quality Monitoring Locations")

    map_data = (
        view[["latitude", "longitude"]]
        .dropna()
        .rename(columns={
            "latitude": "lat",
            "longitude": "lon"
        })
    )

    if not map_data.empty:
        st.map(map_data)
    else:
        st.warning("No valid location coordinates available.")


# ==================================================
# TAB 3: ABOUT
# ==================================================

with tab3:

    st.markdown("## 🌱 About AirGuard")

    st.markdown("""
    <div class="ag-card">
        <h3>What is AirGuard?</h3>
        <p>
        AirGuard is a machine learning based air quality
        classification application designed to analyse
        pollutant concentrations and estimate air quality levels.
        </p>
        <p>
        The system uses environmental measurements and
        weather-related parameters to classify air quality
        into three categories: Safe, Moderate and Toxic / Unhealthy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Technologies Used")

    t1, t2, t3 = st.columns(3)

    t1.markdown(
        '<div class="ag-card"><h3>🐍 Python</h3>'
        '<p>Programming language</p></div>',
        unsafe_allow_html=True
    )

    t2.markdown(
        '<div class="ag-card"><h3>🌳 Random Forest</h3>'
        '<p>Machine learning algorithm</p></div>',
        unsafe_allow_html=True
    )

    t3.markdown(
        '<div class="ag-card"><h3>📊 Streamlit</h3>'
        '<p>Interactive web application</p></div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🎯 Project Objectives")

    st.markdown("""
    - Analyse air quality using pollutant measurements.
    - Classify air quality into three categories.
    - Provide an interactive prediction interface.
    - Visualise environmental data through charts and maps.
    - Demonstrate machine learning applications in environmental monitoring.
    """)

    st.markdown("### 👩‍💻 Project Information")

    st.markdown("""
    **Project:** AirGuard – Air Quality Classification using Machine Learning

    **Model:** Random Forest Classification

    **Platform:** Python and Streamlit

    **Application:** Environmental monitoring and air quality awareness
    """)


# ---------------- FOOTER ----------------

st.markdown("""
<div class="ag-footer">
    🌿 AIRGUARD | Intelligent Air Quality Monitoring System
    <br><br>
    Built using Python, Machine Learning and Streamlit
    <br>
    © 2026 AirGuard Project
</div>
""", unsafe_allow_html=True)
