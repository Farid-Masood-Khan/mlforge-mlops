"""MLForge - Streamlit Demo and Monitoring Dashboard.

Interactive user interface communicating directly with the FastAPI serving endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
API_KEY = "mlforge-dev-secret-key-change-in-prod"
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(
    page_title="MLForge — MLOps Churn Prediction",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ MLForge — End-to-End MLOps Churn Pipeline")
st.markdown(
    "Production-grade ML system featuring Pandera data contracts, MLflow experiment tracking, "
    "FastAPI inference serving, and PSI drift monitoring."
)

tab1, tab2, tab3 = st.tabs(
    ["🎯 Single Prediction", "📂 Batch Inference", "📊 Lineage & Drift Monitor"]
)

# ==============================================================================
# TAB 1: Single Prediction
# ==============================================================================
with tab1:
    st.subheader("Customer Churn Risk Scoring")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure_months = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.5
        )
        total_charges = st.number_input("Total Lifetime Charges ($)", min_value=0.0, value=846.0)

    if st.button("Score Churn Risk", type="primary"):
        payload = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": float(monthly_charges),
            "total_charges": float(total_charges),
        }

        try:
            resp = requests.post(
                f"{API_BASE_URL}/predict", json=payload, headers=HEADERS, timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                prob = result["churn_probability"]
                pred = result["churn_prediction"]
                thresh = result["threshold"]
                meta = result["model"]

                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    risk_label = "HIGH RISK" if pred == 1 else "LOW RISK"
                    badge_color = "red" if pred == 1 else "green"
                    st.metric(
                        label="Predicted Churn Probability",
                        value=f"{prob:.1%}",
                        delta=f"{risk_label} (threshold: {thresh:.2f})",
                        delta_color="inverse" if pred == 1 else "normal",
                    )
                    st.progress(prob)

                with res_col2:
                    st.info(
                        f"**Model Lineage Audit:**\n"
                        f"- **Model:** `{meta['name']}`\n"
                        f"- **Version:** `{meta['version']}` (`{meta['alias']}` alias)\n"
                        f"- **MLflow Run ID:** `{meta['run_id'][:12]}...`\n"
                        f"- **Request ID:** `{result['request_id']}`"
                    )
            else:
                st.error(f"Inference error ({resp.status_code}): {resp.text}")
        except Exception as exc:
            st.error(f"Connection error to API ({API_BASE_URL}): {exc}")

# ==============================================================================
# TAB 2: Batch Inference
# ==============================================================================
with tab2:
    st.subheader("Batch Scoring Pipeline")
    uploaded_file = st.file_uploader("Upload CSV Batch", type=["csv"])

    if st.button("Load Sample Holdout Batch (10 Records)"):
        ref_file = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "reference"
            / "holdout_reference.parquet"
        )
        if ref_file.exists():
            df_sample = pd.read_parquet(ref_file).head(10)
            st.session_state["batch_df"] = df_sample
        else:
            st.warning("Holdout reference file not found. Run training first.")

    if "batch_df" in st.session_state or uploaded_file is not None:
        batch_data: pd.DataFrame = (
            pd.read_csv(uploaded_file) if uploaded_file else st.session_state["batch_df"]
        )
        st.write("Preview of Batch Data:", batch_data.head())

        if st.button("Run Batch Scoring"):
            records = batch_data.to_dict(orient="records")
            # Clean records
            for r in records:
                if "total_charges" in r and pd.isna(r["total_charges"]):
                    r["total_charges"] = None

            try:
                resp = requests.post(
                    f"{API_BASE_URL}/predict/batch", json=records, headers=HEADERS, timeout=10
                )
                if resp.status_code == 200:
                    b_res = resp.json()
                    st.success(
                        f"Scored {b_res['successful']}/{b_res['total']} records successfully. "
                        f"Failed: {b_res['failed']}"
                    )

                    pred_map = {p["index"]: p for p in b_res["predictions"]}
                    scored_df = batch_data.copy()
                    scored_df["churn_probability"] = [
                        pred_map[i]["churn_probability"] if i in pred_map else None
                        for i in range(len(scored_df))
                    ]
                    scored_df["churn_prediction"] = [
                        "CHURN" if pred_map.get(i, {}).get("churn_prediction") == 1 else "RETAIN"
                        for i in range(len(scored_df))
                    ]

                    st.dataframe(scored_df)
                    csv_export = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Scored CSV", csv_export, "scored_batch.csv", "text/csv"
                    )
                else:
                    st.error(f"Batch inference failed: {resp.text}")
            except Exception as exc:
                st.error(f"API Error: {exc}")

# ==============================================================================
# TAB 3: Lineage & Drift Monitor
# ==============================================================================
with tab3:
    st.subheader("Model Performance & Drift Audit")
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("### 🏆 Active Champion Model")
        try:
            info_resp = requests.get(f"{API_BASE_URL}/model/info", headers=HEADERS, timeout=5)
            if info_resp.status_code == 200:
                info = info_resp.json()
                st.json(info)
            else:
                st.warning(f"Could not load model info ({info_resp.status_code})")
        except Exception:
            st.info("API is currently offline. Start API with `./run.ps1 serve` or `make serve`.")

    with m_col2:
        st.markdown("### 🔍 Latest Drift Analysis")
        reports_dir = Path(__file__).resolve().parent.parent / "logs" / "drift_reports"
        json_reports = sorted(reports_dir.glob("*.json")) if reports_dir.exists() else []

        if json_reports:
            latest_report = json_reports[-1]
            with open(latest_report, encoding="utf-8") as f:
                d_data = json.load(f)

            status_style = {
                "pass": "🟢 PASS",
                "warn": "🟡 WARN",
                "fail": "🔴 FAIL",
            }.get(d_data["overall_status"], d_data["overall_status"])

            st.markdown(f"**Report ID:** `{d_data['report_id']}`")
            st.markdown(f"**Overall Drift Status:** **{status_style}**")

            df_features = pd.DataFrame(d_data["features"])
            st.dataframe(df_features, use_container_width=True)
        else:
            st.info("No drift reports generated yet. Run `python -m src.monitoring.report`.")
