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

pca_2d = PCA(n_components=2)

coords_2d = pca_2d.fit_transform(
    tfidf_matrix.toarray()
)

df_pca_2d = pd.DataFrame({
    "titulo": df["titulo"],
    "cluster": df["cluster"],
    "PCA1": coords_2d[:, 0],
    "PCA2": coords_2d[:, 1]
})

df_pca_2d.to_csv(
    "pca_2d.csv",
    index=False,
    encoding="utf-8-sig"
)

pca_3d = PCA(n_components=3)

coords_3d = pca_3d.fit_transform(
    tfidf_matrix.toarray()
)

df_pca_3d = pd.DataFrame({
    "titulo": df["titulo"],
    "cluster": df["cluster"],
    "PCA1": coords_3d[:, 0],
    "PCA2": coords_3d[:, 1],
    "PCA3": coords_3d[:, 2]
})

df_pca_3d.to_csv(
    "pca_3d.csv",
    index=False,
    encoding="utf-8-sig"
)

termos = tfidf_vectorizer.get_feature_names_out()

resultados_clusters = []

for cluster_id in range(k):

    indices = np.argsort(
        kmeans.cluster_centers_[cluster_id]
    )[::-1]

    principais_termos = termos[
        indices[:10]
    ]

    resultados_clusters.append({
        "cluster": cluster_id,
        "tokens_representativos":
            ", ".join(principais_termos)
    })

df_tokens_clusters = pd.DataFrame(
    resultados_clusters
)

df_tokens_clusters.to_csv(
    "tokens_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

similaridade_tfidf = cosine_similarity(
    tfidf_matrix
)

resultados_coesao = []

for cluster_id in sorted(
    df["cluster"].unique()
):

    indices = df.index[
        df["cluster"] == cluster_id
    ].tolist()

    if len(indices) < 2:
        continue

    submatriz = similaridade_tfidf[
        np.ix_(indices, indices)
    ]

    valores = submatriz[
        np.triu_indices_from(
            submatriz,
            k=1
        )
    ]

    resultados_coesao.append({
        "cluster": cluster_id,
        "n_documentos": len(indices),
        "coesao_media": valores.mean()
    })

df_coesao = pd.DataFrame(
    resultados_coesao
)

df_coesao.to_csv(
    "coesao_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

similaridade_centroides = cosine_similarity(
    kmeans.cluster_centers_
)

distancia_centroides = (
    1 - similaridade_centroides
)

df_distancia = pd.DataFrame(
    distancia_centroides,
    index=[
        f"Cluster {i}"
        for i in range(k)
    ],
    columns=[
        f"Cluster {i}"
        for i in range(k)
    ]
)

df_distancia.to_csv(
    "distancia_clusters.csv",
    encoding="utf-8-sig"
)   