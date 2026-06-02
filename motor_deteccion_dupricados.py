# 🧠 AI Duplicate Detection Engine
# Detecta duplicados usando embeddings semánticos (no matching exacto)

import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings
import numpy as np



st.set_page_config(
    page_title="AI Duplicate Detection Engine",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Duplicate Detection Engine")
st.caption(
    "Detecta duplicados semánticos en datasets usando embeddings "
    "en lugar de matching exacto"
)

def cosine_similarity_matrix(embeddings):
    embeddings = np.array(embeddings)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)

    return np.dot(normalized, normalized.T)

# ─────────────────────────────────────────────
# EMBEDDINGS MODEL
# ─────────────────────────────────────────────

def load_embeddings():
    return OllamaEmbeddings(model="mistral")

# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────

def compute_embeddings(texts, embedder):
    return np.array(embedder.embed_documents(texts))


def find_duplicates(df, column, threshold=0.90):
    embedder = load_embeddings()

    texts = df[column].fillna("").astype(str).tolist()
    embeddings = compute_embeddings(texts, embedder)

    #similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix = cosine_similarity_matrix(embeddings)

    duplicates = []
    visited = set()

    for i in range(len(texts)):
        if i in visited:
            continue

        group = [i]

        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] >= threshold:
                group.append(j)
                visited.add(j)

        if len(group) > 1:
            duplicates.append(group)

    return duplicates, similarity_matrix

def export_duplicates(df, groups):
    rows = []

    for gid, group in enumerate(groups):
        for idx in group:
            rows.append({
                "group_id": gid,
                "row_index": idx,
                "record": df.iloc[idx].to_dict()
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown("## 📂 Cargar dataset")

uploaded_file = st.file_uploader("Sube tu CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.write("Vista previa del dataset:")
    st.dataframe(df.head())

    column = st.selectbox("Selecciona columna a analizar", df.columns)

    threshold = st.slider(
        "Umbral de similitud",
        0.5, 1.0, 0.90
    )

    if st.button("🔍 Detectar duplicados"):

        with st.spinner("Calculando embeddings..."):

            groups, sim_matrix = find_duplicates(df, column, threshold)

        st.success(f"Se encontraron {len(groups)} grupos de duplicados")

        # ─────────────────────────────────────────────
        # RESULTADOS
        # ─────────────────────────────────────────────

        st.markdown("## 🔁 Grupos de duplicados")

        for idx, group in enumerate(groups):

            st.markdown(f"### Grupo {idx+1}")

            st.dataframe(df.iloc[group])

        # ─────────────────────────────────────────────
        # MATRIZ DE SIMILITUD
        # ─────────────────────────────────────────────

        st.markdown("## 📊 Matriz de similitud")

        st.write(pd.DataFrame(sim_matrix))

        result_df = export_duplicates(df, groups)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Descargar duplicados (CSV)",
            data=csv,
            file_name="duplicates_report.csv",
            mime="text/csv"
)