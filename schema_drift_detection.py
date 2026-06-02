# 🧠 Schema Drift Detection AI
# Detecta cambios de esquema que pueden romper pipelines downstream

import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="Schema Drift Detection AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Schema Drift Detection AI")
st.caption(
    "Detecta cambios en esquemas de datos que pueden romper "
    "pipelines, ETLs y modelos downstream"
)

# ─────────────────────────────────────────────
# SCHEMA EXTRACTION
# ─────────────────────────────────────────────

def extract_schema(df: pd.DataFrame):

    schema = {}

    for col in df.columns:
        schema[col] = {
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique()),
        }

    return schema


# ─────────────────────────────────────────────
# SCHEMA DRIFT ENGINE
# ─────────────────────────────────────────────

def compare_schemas(old_schema, new_schema):

    drifts = []

    old_cols = set(old_schema.keys())
    new_cols = set(new_schema.keys())

    # ── NEW COLUMNS ───────────────────────────
    added = new_cols - old_cols
    for col in added:
        drifts.append({
            "type": "NEW_COLUMN",
            "column": col,
            "severity": "Medium",
            "description": f"Nueva columna detectada: {col}"
        })

    # ── REMOVED COLUMNS ───────────────────────
    removed = old_cols - new_cols
    for col in removed:
        drifts.append({
            "type": "REMOVED_COLUMN",
            "column": col,
            "severity": "High",
            "description": f"Columna eliminada: {col}"
        })

    # ── TYPE CHANGES ──────────────────────────
    common = old_cols.intersection(new_cols)

    for col in common:

        old_type = old_schema[col]["dtype"]
        new_type = new_schema[col]["dtype"]

        if old_type != new_type:
            drifts.append({
                "type": "TYPE_CHANGE",
                "column": col,
                "severity": "Critical",
                "description":
                    f"Tipo cambiado en '{col}': "
                    f"{old_type} → {new_type}"
            })

        # Null drift
        old_nulls = old_schema[col]["nulls"]
        new_nulls = new_schema[col]["nulls"]

        if abs(new_nulls - old_nulls) > 5:
            drifts.append({
                "type": "NULL_DRIFT",
                "column": col,
                "severity": "Medium",
                "description":
                    f"Cambio importante en nulls en '{col}'"
            })

    return drifts


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown("## 📂 Subir datasets")

col1, col2 = st.columns(2)

with col1:
    baseline_file = st.file_uploader(
        "Dataset BASELINE",
        type=["csv"],
        key="baseline"
    )

with col2:
    new_file = st.file_uploader(
        "Dataset NUEVO",
        type=["csv"],
        key="new"
    )

if baseline_file and new_file:

    baseline_df = pd.read_csv(baseline_file)
    new_df = pd.read_csv(new_file)

    st.markdown("## 📊 Vista previa")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### BASELINE")
        st.dataframe(baseline_df.head())

    with col2:
        st.markdown("### NUEVO")
        st.dataframe(new_df.head())

    if st.button("🧠 Detectar schema drift"):

        baseline_schema = extract_schema(baseline_df)
        new_schema = extract_schema(new_df)

        drifts = compare_schemas(
            baseline_schema,
            new_schema
        )

        st.markdown("## 🚨 Drift detectado")

        if not drifts:
            st.success("No se detectó schema drift 🎉")

        else:

            critical = len([
                d for d in drifts
                if d["severity"] == "Critical"
            ])

            st.metric("Drifts críticos", critical)

            for drift in drifts:

                sev = drift["severity"]

                if sev == "Critical":
                    st.error(
                        f"🔴 [{drift['type']}] "
                        f"{drift['description']}"
                    )

                elif sev == "High":
                    st.warning(
                        f"🟠 [{drift['type']}] "
                        f"{drift['description']}"
                    )

                else:
                    st.info(
                        f"🟡 [{drift['type']}] "
                        f"{drift['description']}"
                    )

        # ─────────────────────────────
        # EXPORT REPORT
        # ─────────────────────────────

        report_df = pd.DataFrame(drifts)

        st.markdown("## 📥 Exportar reporte")

        csv = report_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Descargar reporte de drift",
            data=csv,
            file_name="schema_drift_report.csv",
            mime="text/csv"
        )

        # ─────────────────────────────
        # RAW SCHEMA
        # ─────────────────────────────

        with st.expander("🔍 Ver schemas detectados"):

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Schema baseline")
                st.json(baseline_schema)

            with col2:
                st.markdown("### Schema nuevo")
                st.json(new_schema)