import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Air Quality Surveillance",
    page_icon="🌫️",
    layout="wide"
)

# Pink + orange theme (keeps the existing dashboard layout)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fff0ed 0%, #fff7f0 55%, #ffe8e2 100%);
    }
    h1, h2, h3 { color: #8f3654; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #e95778, #f58a4b);
        color: white;
        border: 0;
        border-radius: 12px;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        color: white;
        border: 1px solid #d94f70;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.72);
        border: 1px solid #ffd4c9;
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD MODEL, METADATA AND DASHBOARD DATA
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "aqi_model.pkl"
    meta_path = BASE_DIR / "model_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.name}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path.name}")

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)

    return model, meta


@st.cache_data
def load_data():
    data_path = BASE_DIR / "dashboard_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dashboard data file not found: {data_path.name}")
    return pd.read_csv(data_path)


try:
    model, meta = load_model()
    data = load_data()
except Exception as exc:
    st.error(f"Unable to load the project files: {exc}")
    st.stop()


# ---------------------------------------------------------
# PREDICTION HELPERS
# ---------------------------------------------------------
def get_model_expected_features(estimator, fallback_columns):
    """Get the feature order recorded by scikit-learn during training."""
    features = getattr(estimator, "feature_names_in_", None)

    if features is not None:
        return list(features)

    # Some pipelines expose feature names on an earlier pipeline step.
    if hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            features = getattr(step, "feature_names_in_", None)
            if features is not None:
                return list(features)

    return list(fallback_columns)


def _walk_estimator(estimator):
    """Yield an estimator and nested estimators from common sklearn wrappers."""
    if estimator is None or estimator == "drop" or estimator == "passthrough":
        return

    yield estimator

    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            yield from _walk_estimator(step)

    elif isinstance(estimator, ColumnTransformer):
        for _, transformer, _ in estimator.transformers:
            yield from _walk_estimator(transformer)

    elif hasattr(estimator, "steps"):
        for _, step in estimator.steps:
            yield from _walk_estimator(step)

    elif hasattr(estimator, "transformers"):
        for _, transformer, _ in estimator.transformers:
            yield from _walk_estimator(transformer)


def get_column_types(estimator):
    """
    Infer numeric/categorical columns from a fitted ColumnTransformer.
    This avoids sending strings into numeric imputers/scalers (the cause
    of the ufunc 'isnan' error in many sklearn pipelines).
    """
    numeric_columns = set()
    categorical_columns = set()

    def inspect(current):
        if isinstance(current, Pipeline):
            for _, step in current.steps:
                inspect(step)
            return

        if isinstance(current, ColumnTransformer):
            fitted_transformers = getattr(current, "transformers_", current.transformers)
            for name, transformer, columns in fitted_transformers:
                if transformer == "drop" or transformer == "passthrough":
                    continue

                selected = []
                if isinstance(columns, str):
                    selected = [columns]
                elif isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                    selected = [c for c in columns if isinstance(c, str)]
                else:
                    # Callable/slice selectors are not safely resolvable here.
                    selected = []

                branch_estimators = list(_walk_estimator(transformer))
                is_categorical_branch = any(
                    isinstance(item, (OneHotEncoder, OrdinalEncoder))
                    for item in branch_estimators
                )

                if is_categorical_branch:
                    categorical_columns.update(selected)
                else:
                    numeric_columns.update(selected)

                inspect(transformer)

    inspect(estimator)
    return numeric_columns, categorical_columns


def prepare_input_for_model(estimator, input_row):
    """Align input columns and their dtypes with the trained model pipeline."""
    expected = get_model_expected_features(estimator, input_row.columns)

    missing = [column for column in expected if column not in input_row.columns]
    extra = [column for column in input_row.columns if column not in expected]

    if missing:
        raise ValueError(
            "The app is missing model feature(s): "
            + ", ".join(missing)
            + ". The app inputs must match the features used during training."
        )

    # Keep only training columns and preserve their exact training order.
    prepared = input_row.loc[:, expected].copy()

    numeric_columns, categorical_columns = get_column_types(estimator)

    # Convert columns selected by the model's numeric preprocessing branch.
    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    # Keep categorical values as strings, while preserving missing values.
    for column in categorical_columns:
        if column in prepared.columns:
            prepared[column] = prepared[column].map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )

    # If the fitted pipeline did not expose column selectors, use the
    # model's recorded pandas dtypes when available.
    if not numeric_columns and not categorical_columns:
        for column in prepared.columns:
            if pd.api.types.is_numeric_dtype(prepared[column]):
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared, extra


def get_prediction_label(prediction):
    """Convert numpy scalar labels to ordinary Python strings."""
    if isinstance(prediction, np.generic):
        prediction = prediction.item()
    return str(prediction)


# ---------------------------------------------------------
# LABEL STYLES
# ---------------------------------------------------------
STYLE = {
    "Safe": (
        "#2ecc71",
        "✅",
        "Air quality is satisfactory. Normal outdoor activity is fine."
    ),
    "Moderate": (
        "#f39c12",
        "⚠️",
        "Air is polluted. Sensitive people (children, elderly, asthma) "
        "should limit long outdoor activity."
    ),
    "Toxic / Unhealthy": (
        "#e74c3c",
        "☠️",
        "Air is highly toxic. Avoid outdoor activity, keep windows closed "
        "and use a mask (N95)."
    ),
    "Toxic": (
        "#e74c3c",
        "☠️",
        "Air quality is unhealthy. Reduce outdoor exposure and follow "
        "local air-quality advisories."
    ),
    "Unhealthy": (
        "#e74c3c",
        "☠️",
        "Air quality is unhealthy. Reduce outdoor exposure and follow "
        "local air-quality advisories."
    )
}


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.title("🌫️ Air Quality Surveillance System")

best_model = meta.get("best_model", "Trained model")
accuracy = meta.get("accuracy")
macro_f1 = meta.get("macro_f1")

caption_parts = [f"ML model: {best_model}"]
if isinstance(accuracy, (int, float)):
    caption_parts.append(f"Test accuracy: {accuracy * 100:.2f}%")
if isinstance(macro_f1, (int, float)):
    caption_parts.append(f"Macro F1: {macro_f1:.3f}")

st.caption("  |  ".join(caption_parts))

tab1, tab2 = st.tabs(["🔍 Predict air quality", "📊 Dashboard"])


# =========================================================
# TAB 1: PREDICTION
# =========================================================
with tab1:
    st.subheader("Enter pollutant readings and weather")

    pollutants = meta.get("pollutants", [])
    seasons = meta.get("seasons", [])

    if not pollutants:
        pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    if not seasons:
        seasons = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]

    c1, c2, c3 = st.columns(3)

    with c1:
        pollutant = st.selectbox("Pollutant", pollutants)
        p_min = st.number_input(
            "Pollutant minimum", min_value=0.0, max_value=300.0, value=30.0
        )
        p_max = st.number_input(
            "Pollutant maximum", min_value=0.0, max_value=500.0, value=120.0
        )
        p_avg = st.number_input(
            "Pollutant average", min_value=0.0, max_value=300.0, value=70.0
        )

    with c2:
        temp = st.slider("Temperature (°C)", 15.0, 40.0, 28.0)
        hum = st.slider("Humidity (%)", 30.0, 95.0, 60.0)
        wind = st.slider("Wind speed (km/h)", 0.0, 30.0, 15.0)

    with c3:
        season = st.selectbox("Season", seasons)
        hour = st.slider("Hour of day", 0, 23, 12)

    if st.button("🌸 Predict Air Quality", type="primary"):
        # Keep the feature names consistent with the training dataset.
        input_row = pd.DataFrame([{
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

        try:
            input_data, extra_features = prepare_input_for_model(model, input_row)

            if extra_features:
                st.warning(
                    "The following app input(s) are not used by the trained model: "
                    + ", ".join(extra_features)
                )

            # Helpful diagnostics can be expanded if another mismatch occurs.
            with st.expander("View prepared model input"):
                st.dataframe(input_data)
                st.write("Input data types:")
                st.write(input_data.dtypes.astype(str))

            prediction_value = model.predict(input_data)[0]
            prediction = get_prediction_label(prediction_value)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = getattr(model, "classes_", None)

                if classes is None:
                    st.info(f"Predicted air-quality class: {prediction}")
                else:
                    confidence_data = pd.Series(
                        probabilities,
                        index=[get_prediction_label(item) for item in classes]
                    )
            else:
                probabilities = None
                confidence_data = None

            color, icon, advice = STYLE.get(
                prediction,
                (
                    "#e74c3c",
                    "ℹ️",
                    "Prediction completed. Please follow local air-quality guidance."
                )
            )

            st.markdown(
                f"""
                <div style="
                    background: {color};
                    padding: 22px;
                    border-radius: 14px;
                    text-align: center;
                    margin-top: 18px;
                ">
                    <h2 style="color: white; margin: 0;">
                        {icon} {prediction}
                    </h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(advice)

            if confidence_data is not None:
                st.write("**Prediction confidence**")
                st.bar_chart(confidence_data)

        except Exception as exc:
            st.error(
                "Prediction could not be completed. "
                "The input features may not match the model's training features."
            )
            st.exception(exc)
            st.warning(
                "If this error continues, the model's training notebook and "
                "model_meta.json are needed to verify the exact feature names, "
                "data types and category encoding."
            )


# =========================================================
# TAB 2: DASHBOARD
# =========================================================
with tab2:
    st.subheader("Dataset overview")

    required_columns = [
        "city", "AQI", "Air_Quality_Class", "pollutant_id",
        "pollutant_avg", "state", "latitude", "longitude"
    ]
    missing_dashboard_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_dashboard_columns:
        st.error(
            "The dashboard dataset is missing required column(s): "
            + ", ".join(missing_dashboard_columns)
        )
    else:
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Records", f"{len(data):,}")
        m2.metric("Cities", data["city"].nunique())
        m3.metric("Average AQI", f"{data['AQI'].mean():.0f}")
        toxic_percentage = (
            data["Air_Quality_Class"] == "Toxic / Unhealthy"
        ).mean() * 100
        m4.metric("Toxic readings", f"{toxic_percentage:.1f}%")

        a, b = st.columns(2)

        with a:
            st.write("**Class distribution**")
            st.bar_chart(data["Air_Quality_Class"].value_counts())

        with b:
            st.write("**Average reading per pollutant**")
            pollutant_average = (
                data.groupby("pollutant_id")["pollutant_avg"].mean()
            )
            st.bar_chart(pollutant_average)

        available_states = sorted(
            data["state"].dropna().astype(str).unique().tolist()
        )
        selected_state = st.selectbox(
            "Filter by state",
            ["All"] + available_states
        )

        view = (
            data if selected_state == "All"
            else data[data["state"].astype(str) == selected_state]
        )

        st.write("**10 most polluted cities (mean AQI)**")
        city_aqi = (
            view.groupby("city")["AQI"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(city_aqi)

        st.write("**Station map**")
        map_data = view[["latitude", "longitude"]].copy()
        map_data["latitude"] = pd.to_numeric(
            map_data["latitude"], errors="coerce"
        )
        map_data["longitude"] = pd.to_numeric(
            map_data["longitude"], errors="coerce"
        )
        map_data = map_data.dropna()

        if not map_data.empty:
            st.map(
                map_data.rename(
                    columns={"latitude": "lat", "longitude": "lon"}
                )
            )
        else:
            st.info("No valid latitude and longitude values are available.")
