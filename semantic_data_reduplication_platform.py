# 🧠 Semantic Data Deduplication Platform
# Fusiona registros duplicados basados en similitud semántica (embeddings)

import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings

st.set_page_config(
    page_title="Semantic Data Deduplication Platform",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Semantic Data Deduplication Platform")
st.caption(
    "Fusiona registros duplicados aunque tengan estructuras o textos diferentes "
    "usando embeddings semánticos"
)



# ─────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────

def load_embedder():
    return OllamaEmbeddings(model="mistral")


def build_embeddings(texts, embedder):
    return np.array(embedder.embed_documents(texts))


def cosine_matrix(embeddings):
    embeddings = np.array(embeddings)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)

    return normalized @ normalized.T


# ─────────────────────────────────────────────
# CLUSTERING LOGIC (SIMPLE UNION FIND)
# ─────────────────────────────────────────────

def cluster_similar(sim_matrix, threshold=0.90):
    n = len(sim_matrix)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    return list(clusters.values())


# ─────────────────────────────────────────────
# MERGE LOGIC (GOLDEN RECORD)
# ─────────────────────────────────────────────

def merge_cluster(df, cluster):
    merged = {}

    for col in df.columns:
        values = df.iloc[cluster][col].dropna().astype(str)

        if len(values) == 0:
            merged[col] = None
        else:
            # estrategia simple: valor más frecuente
            merged[col] = values.mode()[0]

    return merged


def build_merged_dataset(df, clusters):
    rows = []
    for cluster in clusters:
        rows.append(merge_cluster(df, cluster))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

uploaded = st.file_uploader("Sube tu CSV", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)

    st.subheader("📊 Dataset original")
    st.dataframe(df)

    column = st.selectbox(
        "Columna base para semántica (recomendado: name / description)",
        df.columns
    )

    threshold = st.slider("Umbral de similitud", 0.5, 1.0, 0.90)

    if st.button("🧠 Detectar y fusionar duplicados"):

        embedder = load_embedder()

        texts = df[column].fillna("").astype(str).tolist()

        embeddings = build_embeddings(texts, embedder)

        sim_matrix = cosine_matrix(embeddings)

        clusters = cluster_similar(sim_matrix, threshold)

        st.success(f"Clusters encontrados: {len(clusters)}")

        # ─────────────────────────────
        # SHOW CLUSTERS
        # ─────────────────────────────

        st.markdown("## 🔁 Clusters detectados")

        for i, cluster in enumerate(clusters):
            st.markdown(f"### Grupo {i+1}")
            st.dataframe(df.iloc[cluster])

        # ─────────────────────────────
        # MERGED DATASET
        # ─────────────────────────────

        merged_df = build_merged_dataset(df, clusters)

        st.markdown("## 🧬 Dataset fusionado (Golden Records)")
        st.dataframe(merged_df)

        # ─────────────────────────────
        # DOWNLOAD
        # ─────────────────────────────

        csv = merged_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Descargar dataset fusionado",
            data=csv,
            file_name="semantic_deduplicated_dataset.csv",
            mime="text/csv"
        )