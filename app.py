import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

BASE = Path(__file__).parent

st.set_page_config(
    page_title="StudyTrack Analytics",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(BASE / "study_gpa_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv(BASE / "cleaned_student_dataset.csv")

@st.cache_data
def load_metrics():
    return pd.read_csv(BASE / "model_comparison.csv")

@st.cache_data
def load_metadata():
    with open(BASE / "model_metadata.json") as f:
        return json.load(f)

try:
    model = load_model()
    df = load_data()
    metrics = load_metrics()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Files load nahi huin: {exc}")
    st.stop()

features = metadata["feature_columns"]
target = metadata["target"]

st.sidebar.title("📚 StudyTrack")
st.sidebar.caption("Smart Study Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "GPA Predictor", "Dataset Explorer",
     "Model Performance"]
)

st.sidebar.divider()
st.sidebar.caption("Project 01 • Python + Machine Learning")

st.title("StudyTrack Analytics")
st.caption("Explore student lifestyle patterns and GPA predictions.")

if page == "Dashboard":
    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Records", f"{len(df):,}")
    c2.metric("Input Features", len(features))
    c3.metric("Selected Model", metadata["selected_model"])

    st.subheader("Dataset Overview")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        st.dataframe(numeric.describe().T, use_container_width=True)

    if target in df.columns:
        st.subheader("GPA Distribution")
        fig = px.histogram(
            df, x=target, nbins=20,
            title="Distribution of student GPA",
            color_discrete_sequence=["#4385F5"]
        )
        st.plotly_chart(fig, use_container_width=True)

    study_cols = [
        c for c in df.columns
        if c.lower() == "study_hours_per_day"
    ]

    if study_cols and target in df.columns:
        st.subheader("Study Hours vs GPA")
        fig = px.scatter(
            df, x=study_cols[0], y=target,
            title="Reported study hours and GPA",
            trendline="ols",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Dataset observational hai. Correlation causation nahi hoti, "
        "aur predictions guaranteed results nahi hain."
    )

elif page == "GPA Predictor":
    st.subheader("Student GPA Prediction")
    st.write(
        "Apni details enter karo. Model estimated GPA dega, "
        "actual ya guaranteed result nahi."
    )

    with st.form("prediction_form"):
        values = {}

        for col in features:
            if col not in df.columns:
                st.error(f"Required feature missing: {col}")
                st.stop()

            series = df[col]

            if pd.api.types.is_numeric_dtype(series):
                valid = pd.to_numeric(series, errors="coerce").dropna()

                if valid.empty:
                    st.error(f"{col} mein valid numeric data nahi.")
                    st.stop()

                low = float(valid.min())
                high = float(valid.max())
                default = float(valid.median())

                # Dataset ki observed range ke andar input
                if low == high:
                    values[col] = low
                    st.write(f"{col}: {low}")
                elif pd.api.types.is_integer_dtype(series):
                    values[col] = st.number_input(
                        col,
                        min_value=int(low),
                        max_value=int(high),
                        value=max(
                            int(low),
                            min(int(round(default)), int(high))
                        ),
                        step=1
                    )
                else:
                    values[col] = st.number_input(
                        col,
                        min_value=low,
                        max_value=high,
                        value=min(max(default, low), high),
                        step=0.5
                    )
            else:
                choices = series.dropna().astype(str).unique().tolist()

                if not choices:
                    st.error(f"{col} mein options nahi.")
                    st.stop()

                values[col] = st.selectbox(col, choices)

        submitted = st.form_submit_button(
            "Predict GPA",
            type="primary",
            use_container_width=True
        )

    if submitted:
        sample = pd.DataFrame([values], columns=features)

        try:
            prediction = float(model.predict(sample)[0])

            st.metric(
                "Estimated GPA",
                f"{prediction:.2f} / 4.00"
            )

            if prediction < 0 or prediction > 4:
                st.warning(
                    "Prediction expected GPA scale se bahar hai. "
                    "Model output ko carefully interpret karo."
                )

            st.caption(
                "Ye estimate hai, academic decision ya guarantee nahi."
            )

            st.download_button(
                "Download prediction as CSV",
                sample.assign(Predicted_GPA=prediction).to_csv(
                    index=False
                ),
                file_name="studytrack_prediction.csv",
                mime="text/csv"
            )

        except Exception as exc:
            st.error(f"Prediction error: {exc}")

elif page == "Dataset Explorer":
    st.subheader("Cleaned Student Dataset")

    search = st.text_input("Filter rows by text")

    view = df.copy()

    if search.strip():
        mask = view.astype(str).apply(
            lambda col: col.str.contains(
                search, case=False, regex=False, na=False
            )
        ).any(axis=1)
        view = view[mask]

    st.write(f"Showing {len(view):,} of {len(df):,} records")
    st.dataframe(view, use_container_width=True, height=420)

    st.download_button(
        "Download dataset CSV",
        view.to_csv(index=False),
        file_name="cleaned_student_dataset.csv",
        mime="text/csv"
    )

elif page == "Model Performance":
    st.subheader("Model Comparison")

    st.dataframe(metrics, use_container_width=True)

    if "Model" in metrics.columns and "MAE" in metrics.columns:
        fig = px.bar(
            metrics, x="Model", y="MAE",
            title="Mean Absolute Error — lower is better",
            color="Model"
        )
        st.plotly_chart(fig, use_container_width=True)

    if {"Model", "RMSE"}.issubset(metrics.columns):
        fig = px.bar(
            metrics, x="Model", y="RMSE",
            title="Root Mean Squared Error — lower is better",
            color="Model"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "R², MAE aur RMSE ko test-set results ke context mein dekho. "
        "Baseline comparison bhi important hai."
    )

st.divider()
st.caption(
    "StudyTrack • Educational project • "
    "Predictions are estimates, not guarantees."
)
