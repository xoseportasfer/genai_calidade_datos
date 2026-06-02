# 🧼 Intelligent Data Cleaning System
# Limpieza automática de datasets: formatos, valores inconsistentes, missing data

import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(
    page_title="Intelligent Data Cleaning System",
    page_icon="🧼",
    layout="wide"
)

st.title("🧼 Intelligent Data Cleaning System")
st.caption(
    "Limpia datasets automáticamente corrigiendo errores, "
    "formato inconsistente y valores faltantes"
)

# ─────────────────────────────────────────────
# CORE CLEANING FUNCTIONS
# ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return text
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_case(text: str) -> str:
    if pd.isna(text):
        return text
    return str(text).strip().title()


def clean_email(email: str) -> str:
    if pd.isna(email):
        return email
    email = str(email).strip().lower()
    email = re.sub(r"\s+", "", email)
    return email


def clean_phone(phone: str) -> str:
    if pd.isna(phone):
        return phone
    phone = re.sub(r"[^\d+]", "", str(phone))
    return phone


def clean_numeric(value):
    try:
        if pd.isna(value):
            return value
        value = str(value).replace(",", ".")
        return float(re.sub(r"[^\d.]", "", value))
    except:
        return np.nan


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in df.columns:

        # NUMERIC COLUMNS
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

        # TEXT / OBJECT COLUMNS
        else:
            df[col] = df[col].fillna("UNKNOWN")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in df.columns:

        # EMAILS
        if "email" in col.lower():
            df[col] = df[col].apply(clean_email)

        # PHONES
        elif "phone" in col.lower() or "telefono" in col.lower():
            df[col] = df[col].apply(clean_phone)

        # NUMERIC
        elif df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].apply(clean_numeric)

        # TEXT
        else:
            df[col] = df[col].apply(normalize_text)

    # Fix missing values
    df = fill_missing(df)

    # Remove duplicates
    df = remove_duplicates(df)

    return df


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown("## 📂 Carga tu dataset")

uploaded_file = st.file_uploader("Sube CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset original")
    st.dataframe(df)

    if st.button("🧼 Limpiar dataset"):

        cleaned_df = clean_dataset(df)

        st.success("Dataset limpiado correctamente")

        # ─────────────────────────────
        # BEFORE / AFTER
        # ─────────────────────────────

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ❌ Original")
            st.dataframe(df.head())

        with col2:
            st.markdown("### ✅ Limpio")
            st.dataframe(cleaned_df.head())

        # ─────────────────────────────
        # DOWNLOAD
        # ─────────────────────────────

        csv = cleaned_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Descargar dataset limpio",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )