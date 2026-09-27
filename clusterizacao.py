import pandas as pd
import ast
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# CARREGAR DADOS
# ==========================================================

df = pd.read_csv(
    "noticias_ge_processado.csv",
    encoding="utf-8-sig"
)

df["tokens_sem_stopwords"] = df[
    "tokens_sem_stopwords"
].apply(ast.literal_eval)

# ==========================================================
# TF-IDF
# ==========================================================

tfidf_vectorizer = TfidfVectorizer(
    analyzer=lambda tokens: tokens
)

tfidf_matrix = tfidf_vectorizer.fit_transform(
    df["tokens_sem_stopwords"]
)

k = 10

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(
    tfidf_matrix
)

k = 10

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(
    tfidf_matrix
)

df.to_csv(
    "noticias_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# PCA 2D E 3D DOS CLUSTERS
# ==========================================================

from sklearn.decomposition import PCA
import plotly.express as px


# ==========================================================
# PCA EM 2 DIMENSÕES
# ==========================================================

pca_2d = PCA(
    n_components=2
)

coordenadas_2d = pca_2d.fit_transform(
    tfidf_matrix.toarray()
)


# Criar DataFrame para visualização
df_pca_2d = pd.DataFrame({
    "id": df["id"],
    "titulo": df["titulo"],
    "texto": df["texto"],
    "cluster": df["cluster"].astype(str),
    "PCA1": coordenadas_2d[:, 0],
    "PCA2": coordenadas_2d[:, 1]
})


# Salvar coordenadas 2D
df_pca_2d.to_csv(
    "pca_2d.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# GRÁFICO 2D
# ==========================================================

fig_2d = px.scatter(

    df_pca_2d,

    x="PCA1",

    y="PCA2",

    color="cluster",

    hover_name="titulo",

    hover_data={

        "id": True,

        "texto": True,

        "cluster": True,

        "PCA1": False,

        "PCA2": False

    },

    title="Clusters das notícias — PCA em 2 dimensões"

)


# Tamanho dos pontos
fig_2d.update_traces(

    marker=dict(
        size=10
    )

)


# Configuração dos eixos
fig_2d.update_layout(

    xaxis_title="Componente principal 1",

    yaxis_title="Componente principal 2",

    legend_title="Cluster",

    width=1200,

    height=750

)


# ==========================================================
# SALVAR HTML INTERATIVO — PCA 2D
# ==========================================================

fig_2d.write_html(

    "clusters_pca_2d.html",

    auto_open=False

)


# Mostrar gráfico
fig_2d.show()


# ==========================================================
# PCA EM 3 DIMENSÕES
# ==========================================================

pca_3d = PCA(

    n_components=3

)


coordenadas_3d = pca_3d.fit_transform(

    tfidf_matrix.toarray()

)


# Criar DataFrame para visualização
df_pca_3d = pd.DataFrame({

    "id": df["id"],

    "titulo": df["titulo"],

    "texto": df["texto"],

    "cluster": df["cluster"].astype(str),

    "PCA1": coordenadas_3d[:, 0],

    "PCA2": coordenadas_3d[:, 1],

    "PCA3": coordenadas_3d[:, 2]

})


# Salvar coordenadas 3D
df_pca_3d.to_csv(

    "pca_3d.csv",

    index=False,

    encoding="utf-8-sig"

)


# ==========================================================
# GRÁFICO 3D
# ==========================================================

fig_3d = px.scatter_3d(

    df_pca_3d,

    x="PCA1",

    y="PCA2",

    z="PCA3",

    color="cluster",

    hover_name="titulo",

    hover_data={

        "id": True,

        "texto": True,

        "cluster": True,

        "PCA1": False,

        "PCA2": False,

        "PCA3": False

    },

    title="Clusters das notícias — PCA em 3 dimensões"

)


# Tamanho dos pontos
fig_3d.update_traces(

    marker=dict(
        size=6
    )

)


# Configuração dos eixos
fig_3d.update_layout(

    scene=dict(

        xaxis_title="Componente principal 1",

        yaxis_title="Componente principal 2",

        zaxis_title="Componente principal 3"

    ),

    legend_title="Cluster",

    width=1200,

    height=800

)


# ==========================================================
# SALVAR HTML INTERATIVO — PCA 3D
# ==========================================================

fig_3d.write_html(

    "clusters_pca_3d.html",

    auto_open=False

)


# Mostrar gráfico
fig_3d.show()   