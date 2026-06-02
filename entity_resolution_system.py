# 🧠 Entity Resolution System (Golden Record AI)
# Unifica identidades dispersas y genera un “single source of truth”

import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher

st.set_page_config(
    page_title="Entity Resolution System (Golden Record AI)",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Entity Resolution System (Golden Record AI)")
st.caption(
    "Construye un single source of truth unificando identidades "
    "dispersas entre múltiples registros"
)

# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


# ─────────────────────────────────────────────
# SIMILARITY ENGINE
# ─────────────────────────────────────────────

def similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


def calculate_entity_score(row1, row2):

    score = 0.0

    # Name similarity
    if "name" in row1 and "name" in row2:
        score += similarity(row1["name"], row2["name"]) * 0.5

    # Email similarity
    if "email" in row1 and "email" in row2:
        score += similarity(row1["email"], row2["email"]) * 0.3

    # Phone similarity
    if "phone" in row1 and "phone" in row2:
        score += similarity(row1["phone"], row2["phone"]) * 0.2

    return score


# ─────────────────────────────────────────────
# ENTITY CLUSTERING
# ─────────────────────────────────────────────

def resolve_entities(df, threshold=0.85):

    visited = set()
    groups = []

    for i in range(len(df)):

        if i in visited:
            continue

        current_group = [i]
        visited.add(i)

        for j in range(i + 1, len(df)):

            if j in visited:
                continue

            score = calculate_entity_score(
                df.iloc[i],
                df.iloc[j]
            )

            if score >= threshold:
                current_group.append(j)
                visited.add(j)

        groups.append(current_group)

    return groups


# ─────────────────────────────────────────────
# GOLDEN RECORD GENERATION
# ─────────────────────────────────────────────

def build_golden_record(df, group):

    records = df.iloc[group]

    golden = {}

    for col in df.columns:

        values = (
            records[col]
            .dropna()
            .astype(str)
        )

        if len(values) == 0:
            golden[col] = None

        else:
            # valor más frecuente
            golden[col] = values.mode()[0]

    golden["source_records"] = len(group)

    return golden


def build_master_dataset(df, groups):

    master = []

    for group in groups:
        master.append(
            build_golden_record(df, group)
        )

    return pd.DataFrame(master)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown("## 📂 Subir dataset")

uploaded = st.file_uploader(
    "Sube CSV",
    type=["csv"]
)

if uploaded:

    df = pd.read_csv(uploaded)

    st.subheader("📊 Dataset original")
    st.dataframe(df)

    threshold = st.slider(
        "Threshold de matching",
        0.50,
        1.00,
        0.85
    )

    if st.button("🧠 Resolver entidades"):

        groups = resolve_entities(df, threshold)

        st.success(
            f"Entidades maestras generadas: {len(groups)}"
        )

        # ─────────────────────────────
        # SHOW CLUSTERS
        # ─────────────────────────────

        st.markdown("## 🔗 Clusters detectados")

        for idx, group in enumerate(groups):

            with st.expander(
                f"Entidad {idx+1} — "
                f"{len(group)} registros",
                expanded=False
            ):
                st.dataframe(df.iloc[group])

        # ─────────────────────────────
        # GOLDEN RECORDS
        # ─────────────────────────────

        master_df = build_master_dataset(df, groups)

        st.markdown("## 🏆 Golden Records")

        st.dataframe(master_df)

        # ─────────────────────────────
        # METRICS
        # ─────────────────────────────

        original = len(df)
        deduplicated = len(master_df)

        reduction = round(
            (1 - deduplicated / original) * 100,
            2
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Registros originales",
                original
            )

        with col2:
            st.metric(
                "Golden records",
                deduplicated
            )

        with col3:
            st.metric(
                "Reducción duplicados",
                f"{reduction}%"
            )

        # ─────────────────────────────
        # EXPORT
        # ─────────────────────────────

        csv = master_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Descargar Golden Records",
            data=csv,
            file_name="golden_records.csv",
            mime="text/csv"
        )