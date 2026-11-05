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
