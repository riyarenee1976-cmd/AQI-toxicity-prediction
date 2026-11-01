import streamlit as st
import pandas as pd
import json
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AirGuard | Air Quality Surveillance",
    page_icon="🌸",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("aqi_model.pkl")

    with open("model_meta.json", "r") as file:
        meta = json.load(file)

    return model, meta


@st.cache_data
def load_data():
    return pd.read_csv("dashboard_data.csv")


model, meta = load_model()
data = load_data()


# =========================================================
# COLOUR THEME
# =========================================================

STYLE = {
    "Safe": (
        "#249B72",
        "🟢",
        "Air quality is satisfactory. Normal outdoor activity is fine."
    ),

    "Moderate": (
        "#F28C55",
        "🟠",
        "Air is moderately polluted. Sensitive individuals should limit prolonged outdoor activity."
    ),

    "Toxic / Unhealthy": (
        "#D94867",
        "🔴",
        "Air quality is unhealthy. Reduce outdoor exposure and follow local health guidance."
    )
}


# =========================================================
# CUSTOM CSS – PINK AND ORANGE THEME
# =========================================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background: linear-gradient(
        135deg,
        #fff5f0 0%,
        #ffe8e8 50%,
        #fff0df 100%
    );
    color: #492b35;
}

/* Main content */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */

h1, h2, h3 {
    color: #542c3d !important;
    font-weight: 700 !important;
}

/* Paragraphs */

p, label, .stMarkdown {
    color: #654451;
}

/* Header */

.ag-header {
    background: linear-gradient(
        110deg,
        #e85d75,
        #f28c55
    );
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 22px rgba(210, 90, 110, 0.15);
}

.ag-header h1 {
    color: white !important;
    font-size: 38px;
    margin-bottom: 5px;
}

.ag-header p {
    color: #fff7f2 !important;
    font-size: 16px;
}

/* Cards */

.ag-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid #f6c8c5;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(190, 100, 110, 0.08);
    margin-bottom: 15px;
}

/* Prediction result */

.ag-result {
    background: #fff0e8;
    border: 2px solid #f6b7a9;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
    text-align: center;
}

/* Metric cards */

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #f6c8c5;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(190, 100, 110, 0.06);
}

[data-testid="stMetricValue"] {
    color: #d94e70;
    font-weight: 700;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        100deg,
        #e85d75,
        #f28c55
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 25px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(
        100deg,
        #d94e70,
        #e87845
    );
    color: white;
    border: none;
    transform: translateY(-2px);
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: #fff0e8;
    border-radius: 10px;
    color: #754454;
    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {
    background: #e85d75 !important;
    color: white !important;
}

/* Input boxes */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #fffaf7;
    border-radius: 10px;
}

/* Footer */

.ag-footer {
    text-align: center;
    color: #9b7180;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="ag-header">

    <h1>🌸 AirGuard</h1>

    <p>
        Intelligent Air Quality Surveillance and
        Toxicity Prediction System
    </p>

    <p>
        Monitor air pollution. Predict air quality.
        Make healthier decisions.
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_pollutant_options():
    pollutants = meta.get("pollutants", [])

    if isinstance(pollutants, dict):
        return list(pollutants.keys())

    if isinstance(pollutants, list) and pollutants:
        if isinstance(pollutants[0], dict):
            return [
                item.get("name", str(item))
                for item in pollutants
            ]

        return pollutants

    return ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]


def get_pollutant_id(pollutant):
    pollutants = meta.get("pollutants", [])

    if isinstance(pollutants, dict):
        value = pollutants.get(pollutant)

        if isinstance(value, dict):
            return value.get("id", 0)

        if value is not None:
            return value

    if isinstance(pollutants, list):
        for index, item in enumerate(pollutants):
            if isinstance(item, dict):
                if item.get("name") == pollutant:
                    return item.get("id", index)

            elif item == pollutant:
                return index

    return 0


def show_result(result, confidence=None):
    result = str(result)

    colour, emoji, message = STYLE.get(
        result,
        ("#D94867", "⚪", "Prediction completed.")
    )

    st.markdown(f"""
    <div class="ag-result">

        <h2 style="color:{colour};">
            {emoji} {result}
        </h2>

        <p style="font-size:16px;">
            {message}
        </p>

    </div>
    """, unsafe_allow_html=True)

    if confidence is not None:
        st.write("### Prediction confidence")

        st.progress(float(confidence))

        st.write(f"**Confidence: {confidence * 100:.2f}%**")


# =========================================================
# NAVIGATION TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Predict Air Quality",
    "📊 Analytics Dashboard",
    "🌸 About AirGuard"
])


# =========================================================
# TAB 1 – AIR QUALITY PREDICTION
# =========================================================

with tab1:

    st.markdown("## 🔍 Air Quality Prediction")

    st.write(
        "Enter the pollutant and environmental measurements "
        "to predict the air quality category."
    )

    st.markdown('<div class="ag-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    pollutant_options = get_pollutant_options()

    with col1:

        st.subheader("Pollutant Information")

        pollutant = st.selectbox(
            "Select Pollutant",
            pollutant_options
        )

        pollutant_min = st.number_input(
            "Minimum Pollutant Value",
            min_value=0.0,
            value=10.0,
            step=0.1
        )

        pollutant_max = st.number_input(
            "Maximum Pollutant Value",
            min_value=0.0,
            value=50.0,
            step=0.1
        )

        pollutant_avg = st.number_input(
            "Average Pollutant Value",
            min_value=0.0,
            value=25.0,
            step=0.1
        )

    with col2:

        st.subheader("Environmental Information")

        temperature = st.number_input(
            "Temperature (°C)",
            value=28.0,
            step=0.1
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.1
        )

        wind_speed = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            value=10.0,
            step=0.1
        )

        hour = st.slider(
            "Hour of the Day",
            min_value=0,
            max_value=23,
            value=12
        )

        season = st.selectbox(
            "Season",
            ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    if st.button("🌸 Predict Air Quality", use_container_width=True):

        if pollutant_min > pollutant_max:
            st.error(
                "Minimum pollutant value cannot be greater "
                "than maximum pollutant value."
            )

        elif not pollutant_min <= pollutant_avg <= pollutant_max:
            st.error(
                "Average pollutant value should be between "
                "the minimum and maximum values."
            )

        else:

            # Input features must match the training features.

            input_data = pd.DataFrame([{
                "pollutant_min": pollutant_min,
                "pollutant_max": pollutant_max,
                "pollutant_avg": pollutant_avg,
                "Temperature_C": temperature,
                "Humidity_%": humidity,
                "Wind_Speed_kmh": wind_speed,
                "Hour": hour,
                "pollutant_id": get_pollutant_id(pollutant),
                "Season": season
            }])

            try:

                prediction = model.predict(input_data)[0]

                confidence = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)[0]
                    confidence = max(probabilities)

                st.markdown("### Prediction Result")

                show_result(prediction, confidence)

            except Exception as error:

                st.error(
                    "Prediction could not be completed. "
                    "Please check that the input features "
                    "match the model's training features."
                )

                st.exception(error)


# =========================================================
# TAB 2 – ANALYTICS DASHBOARD
# =========================================================

with tab2:

    st.markdown("## 📊 Air Quality Analytics Dashboard")

    st.write(
        "Explore air quality records, pollution levels "
        "and geographical distribution."
    )

    # Check available columns

    required_columns = [
        "AQI",
        "Air_Quality_Class",
        "city"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        st.warning(
            "The dashboard is missing these columns: "
            + ", ".join(missing_columns)
        )

    else:

        # ---------------------------------------------
        # KEY METRICS
        # ---------------------------------------------

        total_records = len(data)

        average_aqi = data["AQI"].mean()

        highest_aqi = data["AQI"].max()

        toxic_records = data[
            data["Air_Quality_Class"].astype(str).str.contains(
                "toxic|unhealthy",
                case=False,
                na=False
            )
        ]

        toxic_percentage = (
            len(toxic_records) / total_records * 100
            if total_records > 0 else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Records",
            f"{total_records:,}"
        )

        col2.metric(
            "Average AQI",
            f"{average_aqi:.2f}"
        )

        col3.metric(
            "Highest AQI",
            f"{highest_aqi:.2f}"
        )

        col4.metric(
            "Toxic / Unhealthy",
            f"{toxic_percentage:.1f}%"
        )

        st.divider()

        # ---------------------------------------------
        # AIR QUALITY DISTRIBUTION
        # ---------------------------------------------

        st.subheader("🌸 Air Quality Class Distribution")

        class_counts = data["Air_Quality_Class"].value_counts()

        col1, col2 = st.columns(2)

        with col1:

            st.bar_chart(class_counts)

        with col2:

            st.write("### Class-wise Records")

            st.dataframe(
                class_counts.rename("Number of Records"),
                use_container_width=True
            )

        st.divider()

        # ---------------------------------------------
        # AQI ANALYSIS
        # ---------------------------------------------

        st.subheader("📈 AQI Analysis")

        if "state" in data.columns:

            states = sorted(
                data["state"].dropna().unique().tolist()
            )

            selected_states = st.multiselect(
                "Filter by State",
                states,
                default=states
            )

            filtered_data = data[
                data["state"].isin(selected_states)
            ]

        else:

            filtered_data = data.copy()

        if not filtered_data.empty:

            city_aqi = (
                filtered_data.groupby("city")["AQI"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )

            st.write("### Top 10 Cities with Highest Average AQI")

            st.bar_chart(city_aqi)

            st.write("### AQI Dataset")

            st.dataframe(
                filtered_data,
                use_container_width=True,
                height=350
            )

        else:

            st.info("No records available for the selected filter.")

        st.divider()

        # ---------------------------------------------
        # POLLUTANT ANALYSIS
        # ---------------------------------------------

        if "pollutant_id" in data.columns and "pollutant_avg" in data.columns:

            st.subheader("🌫️ Pollutant Analysis")

            pollutant_analysis = (
                data.groupby("pollutant_id")["pollutant_avg"]
                .mean()
                .sort_values(ascending=False)
            )

            st.bar_chart(pollutant_analysis)

        st.divider()

        # ---------------------------------------------
        # GEOGRAPHICAL DISTRIBUTION
        # ---------------------------------------------

        st.subheader("📍 Air Quality Monitoring Locations")

        if "latitude" in data.columns and "longitude" in data.columns:

            map_data = filtered_data[
                ["latitude", "longitude"]
            ].copy()

            map_data = map_data.dropna()

            map_data = map_data.rename(columns={
                "latitude": "lat",
                "longitude": "lon"
            })

            if not map_data.empty:

                st.map(map_data)

            else:

                st.info("No valid geographical coordinates available.")

        else:

            st.info(
                "Latitude and longitude columns are not available "
                "in the dataset."
            )


# =========================================================
# TAB 3 – ABOUT AIRGUARD
# =========================================================

with tab3:

    st.markdown("## 🌸 About AirGuard")

    st.markdown("""
    <div class="ag-card">

        <h2 style="color:#d94e70;">
            🌍 Intelligent Air Quality Monitoring
        </h2>

        <p>
            AirGuard is a machine learning-based air quality
            surveillance and toxicity prediction system.
        </p>

        <p>
            The application analyses pollutant concentrations
            and environmental parameters to predict air quality
            categories and help users understand pollution levels.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Project Objectives")

    st.markdown("""
    - Predict air quality using machine learning.
    - Classify air conditions into Safe, Moderate and Toxic / Unhealthy.
    - Analyse pollutant concentrations.
    - Visualise air quality data through an interactive dashboard.
    - Support awareness of environmental pollution.
    """)

    st.markdown("### 🤖 Machine Learning Model")

    st.markdown("""
    <div class="ag-card">

        <h3 style="color:#e85d75;">
            Random Forest Classifier
        </h3>

        <p>
            AirGuard uses a Random Forest classification model
            to predict air quality categories from pollutant
            and environmental measurements.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Technologies Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Programming**

        - Python
        - Pandas
        - NumPy
        """)

    with col2:
        st.markdown("""
        **Machine Learning**

        - Scikit-learn
        - Random Forest
        - Joblib
        """)

    with col3:
        st.markdown("""
        **Application**

        - Streamlit
        - Data Visualisation
        - GitHub
        """)

    st.markdown("### 🌱 Our Goal")

    st.write(
        "To make air quality information easier to understand "
        "through machine learning, data analysis and interactive "
        "visualisation."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="ag-footer">

    <p>
        🌸 AirGuard | Intelligent Air Quality Surveillance
    </p>

    <p>
        Machine Learning Mini Project
    </p>

</div>
""", unsafe_allow_html=True)
