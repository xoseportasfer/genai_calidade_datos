# 🧠 Data Consistency Validator AI
# Detecta inconsistencias lógicas entre columnas, reglas de negocio y fuentes de datos

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Data Consistency Validator AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Data Consistency Validator AI")
st.caption(
    "Detecta inconsistencias lógicas entre columnas, reglas de negocio "
    "y datos distribuidos"
)

# ─────────────────────────────────────────────
# CORE CONSISTENCY RULES ENGINE
# ─────────────────────────────────────────────

def validate_consistency(df: pd.DataFrame):
    issues = []

    # ── RULE 1: Age must be realistic ─────────────────────────
    if "age" in df.columns:
        invalid_age = df[(df["age"] < 0) | (df["age"] > 120)]
        if not invalid_age.empty:
            issues.append({
                "rule": "Edad válida (0-120)",
                "count": len(invalid_age),
                "rows": invalid_age.index.tolist()
            })

    # ── RULE 2: Email must contain @ ─────────────────────────
    if "email" in df.columns:
        invalid_email = df[~df["email"].astype(str).str.contains("@", na=False)]
        if not invalid_email.empty:
            issues.append({
                "rule": "Email debe contener '@'",
                "count": len(invalid_email),
                "rows": invalid_email.index.tolist()
            })

    # ── RULE 3: Name and email consistency (same user duplication hint) ──
    if "name" in df.columns and "email" in df.columns:
        duplicates = df[df.duplicated(subset=["email"], keep=False)]
        if not duplicates.empty:
            issues.append({
                "rule": "Emails duplicados (posible conflicto de identidad)",
                "count": len(duplicates),
                "rows": duplicates.index.tolist()
            })

    # ── RULE 4: Price must be positive ────────────────────────
    if "price" in df.columns:
        invalid_price = df[df["price"] <= 0]
        if not invalid_price.empty:
            issues.append({
                "rule": "Precio debe ser > 0",
                "count": len(invalid_price),
                "rows": invalid_price.index.tolist()
            })

    # ── RULE 5: Country consistency normalization check ──────
    if "country" in df.columns:
        inconsistent = df[df["country"].astype(str).str.lower() != df["country"].astype(str)]
        if not inconsistent.empty:
            issues.append({
                "rule": "Inconsistencia de formato en country (mayúsculas/minúsculas)",
                "count": len(inconsistent),
                "rows": inconsistent.index.tolist()
            })

    return issues


def fix_basic_consistency(df: pd.DataFrame):
    df = df.copy()

    # Normalize text columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # Fix country casing
    if "country" in df.columns:
        df["country"] = df["country"].str.title()

    # Clamp age
    if "age" in df.columns:
        df["age"] = df["age"].clip(0, 120)

    # Remove invalid emails (optional correction)
    if "email" in df.columns:
        df["email"] = df["email"].fillna("unknown@example.com")

    # Ensure positive prices
    if "price" in df.columns:
        df["price"] = df["price"].abs()

    return df


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown("## 📂 Cargar dataset")

file = st.file_uploader("Sube CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("📊 Dataset original")
    st.dataframe(df)

    if st.button("🧠 Validar consistencia"):

        issues = validate_consistency(df)

        st.markdown("## 🚨 Inconsistencias detectadas")

        if not issues:
            st.success("No se encontraron inconsistencias 🎉")
        else:
            for issue in issues:
                st.error(
                    f"🔴 {issue['rule']} | "
                    f"{issue['count']} filas afectadas"
                )
                st.write("Filas:", issue["rows"])

        # ─────────────────────────────
        # FIXED DATASET
        # ─────────────────────────────

        st.markdown("## 🧼 Dataset corregido")

        fixed_df = fix_basic_consistency(df)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ❌ Original")
            st.dataframe(df.head())

        with col2:
            st.markdown("### ✅ Corregido")
            st.dataframe(fixed_df.head())

        # ─────────────────────────────
        # DOWNLOAD
        # ─────────────────────────────

        csv = fixed_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Descargar dataset corregido",
            data=csv,
            file_name="validated_dataset.csv",
            mime="text/csv"
        )