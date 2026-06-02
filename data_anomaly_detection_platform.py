# 🧠 Data Anomaly Detection Platform
# Detecta outliers y valores anómalos en datasets (numéricos y categóricos)

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Data Anomaly Detection Platform",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Data Anomaly Detection Platform")
st.caption(
    "Detecta valores raros, outliers estadísticos y anomalías en datasets masivos"
)

# ─────────────────────────────────────────────
# ANOMALY DETECTION ENGINE
# ─────────────────────────────────────────────

def detect_outliers_iqr(df, col, factor=1.5):

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    anomalies = df[(df[col] < lower) | (df[col] > upper)]

    return anomalies, lower, upper


def detect_zscore_outliers(df, col, threshold=3):

    mean = df[col].mean()
    std = df[col].std()

    if std == 0:
        return pd.DataFrame(), mean, std

    zscores = (df[col] - mean) / std
    anomalies = df[np.abs(zscores) > threshold]

    return anomalies, mean, std


def detect_categorical_anomalies(df, col, min_freq=0.02):

    freq = df[col].value_counts(normalize=True)
    rare_values = freq[freq < min_freq].index

    anomalies = df[df[col].isin(rare_values)]

    return anomalies


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

uploaded = st.file_uploader("Sube tu CSV", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)

    st.subheader("📊 Dataset")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    method = st.selectbox(
        "Método de detección",
        [
            "IQR (Interquartile Range)",
            "Z-Score",
            "Rare Category Detection"
        ]
    )

    if st.button("🧠 Detectar anomalías"):

        st.markdown("## 🚨 Anomalías detectadas")

        total_anomalies = 0

        # ─────────────────────────────
        # NUMERIC ANOMALIES
        # ─────────────────────────────

        if method in ["IQR (Interquartile Range)", "Z-Score"]:

            for col in numeric_cols:

                if method == "IQR (Interquartile Range)":
                    anomalies, low, high = detect_outliers_iqr(df, col)
                    reason = f"IQR bounds [{low:.2f}, {high:.2f}]"

                else:
                    anomalies, mean, std = detect_zscore_outliers(df, col)
                    reason = f"Z-score > 3 (mean={mean:.2f})"

                if not anomalies.empty:

                    total_anomalies += len(anomalies)

                    st.error(
                        f"🔴 Columna '{col}' → {len(anomalies)} anomalías ({reason})"
                    )

                    st.dataframe(anomalies)

        # ─────────────────────────────
        # CATEGORICAL ANOMALIES
        # ─────────────────────────────

        if method == "Rare Category Detection":

            for col in categorical_cols:

                anomalies = detect_categorical_anomalies(df, col)

                if not anomalies.empty:

                    total_anomalies += len(anomalies)

                    st.warning(
                        f"🟡 Columna '{col}' → valores raros detectados"
                    )

                    st.dataframe(anomalies)

        # ─────────────────────────────
        # SUMMARY
        # ─────────────────────────────

        st.markdown("## 📊 Resumen")

        st.metric("Total anomalías detectadas", total_anomalies)

        if total_anomalies == 0:
            st.success("No se detectaron anomalías 🎉")