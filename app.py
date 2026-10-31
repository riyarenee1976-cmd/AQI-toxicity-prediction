import streamlit as st
import pandas as pd
import joblib, json

st.set_page_config(page_title="Air Quality Surveillance", page_icon="🌫️", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("aqi_model.pkl"), json.load(open("model_meta.json"))

@st.cache_data
def load_data():
    return pd.read_csv("dashboard_data.csv")

model, meta = load_model()
data = load_data()

STYLE = {
    "Safe":              ("#2ecc71", "✅", "Air quality is satisfactory. Normal outdoor activity is fine."),
    "Moderate":          ("#f39c12", "⚠️", "Air is polluted. Sensitive people (children, elderly, asthma) should limit long outdoor activity."),
    "Toxic / Unhealthy": ("#e74c3c", "☠️", "Air is highly toxic. Avoid outdoor activity, keep windows closed and use a mask (N95)."),
}

st.title("🌫️ Air Quality Surveillance System")
st.caption(f"ML model: {meta['best_model']}  |  Test accuracy: {meta['accuracy']*100:.2f}%  |  Macro F1: {meta['macro_f1']:.3f}")

tab1, tab2 = st.tabs(["🔍 Predict air quality", "📊 Dashboard"])

# ---------------- TAB 1 : PREDICT ----------------
with tab1:
    st.subheader("Enter pollutant readings and weather")
    c1, c2, c3 = st.columns(3)
    with c1:
        pollutant = st.selectbox("Pollutant", meta["pollutants"])
        p_min = st.number_input("Pollutant minimum", 0.0, 300.0, 30.0)
        p_max = st.number_input("Pollutant maximum", 0.0, 500.0, 120.0)
        p_avg = st.number_input("Pollutant average", 0.0, 300.0, 70.0)
    with c2:
        temp = st.slider("Temperature (°C)", 15.0, 40.0, 28.0)
        hum = st.slider("Humidity (%)", 30.0, 95.0, 60.0)
        wind = st.slider("Wind speed (km/h)", 0.0, 30.0, 15.0)
    with c3:
        season = st.selectbox("Season", meta["seasons"])
        hour = st.slider("Hour of day", 0, 23, 12)

    if st.button("Check air quality", type="primary"):
        row = pd.DataFrame([{
            "pollutant_min": p_min, "pollutant_max": p_max, "pollutant_avg": p_avg,
            "Temperature_C": temp, "Humidity_%": hum, "Wind_Speed_kmh": wind, "Hour": hour,
            "pollutant_id": pollutant, "Season": season}])
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0]
        color, icon, advice = STYLE[pred]

        st.markdown(
            f"<div style='background:{color};padding:22px;border-radius:12px;text-align:center'>"
            f"<h1 style='color:white;margin:0'>{icon} {pred}</h1></div>", unsafe_allow_html=True)
        st.info(advice)

        st.write("**Prediction confidence**")
        st.bar_chart(pd.Series(proba, index=model.classes_))

# ---------------- TAB 2 : DASHBOARD ----------------
with tab2:
    st.subheader("Dataset overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", f"{len(data):,}")
    m2.metric("Cities", data["city"].nunique())
    m3.metric("Average AQI", f"{data['AQI'].mean():.0f}")
    m4.metric("Toxic readings", f"{(data['Air_Quality_Class']=='Toxic / Unhealthy').mean()*100:.1f}%")

    a, b = st.columns(2)
    with a:
        st.write("**Class distribution**")
        st.bar_chart(data["Air_Quality_Class"].value_counts())
    with b:
        st.write("**Average reading per pollutant**")
        st.bar_chart(data.groupby("pollutant_id")["pollutant_avg"].mean())

    city = st.selectbox("Filter by state", ["All"] + sorted(data["state"].unique()))
    view = data if city == "All" else data[data["state"] == city]

    st.write("**10 most polluted cities (mean AQI)**")
    st.bar_chart(view.groupby("city")["AQI"].mean().sort_values(ascending=False).head(10))

    st.write("**Station map**")
    st.map(view[["latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"}))
