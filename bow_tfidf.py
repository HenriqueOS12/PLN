# pip install scikit-learn
# pip install plotly
import pandas as pd
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

# Arquivo de relatório
relatorio = open(
    "noticias_ge_tfidf_bow.txt",
    "w",
    encoding="utf-8"
)

def escrever(texto=""):
    relatorio.write(str(texto) + "\n")

# ==========================================================
# 1. CARREGAR DADOS PROCESSADOS
# ==========================================================

escrever("=" * 50)
escrever("CARREGANDO DADOS")
escrever("=" * 50)

df = pd.read_csv(
    "noticias_ge_processado.csv",
    encoding="utf-8-sig"
)

# O CSV salva a lista de tokens como texto.
# Aqui transformamos novamente em lista Python.
df["tokens_sem_stopwords"] = df[
    "tokens_sem_stopwords"
].apply(ast.literal_eval)

escrever(f"Documentos encontrados: {len(df)}")


# ==========================================================
# 2. BAG OF WORDS
# ==========================================================

escrever()
escrever("=" * 50)
escrever("BAG OF WORDS")
escrever("=" * 50)

bow_vectorizer = CountVectorizer(
    analyzer=lambda tokens: tokens
)

bow_matrix = bow_vectorizer.fit_transform(
    df["tokens_sem_stopwords"]
)

bow_features = bow_vectorizer.get_feature_names_out()

escrever(f"Documentos: {bow_matrix.shape[0]}")
escrever(f"Palavras no vocabulário: {bow_matrix.shape[1]}")
escrever(f"Matriz: {bow_matrix.shape[0]} x {bow_matrix.shape[1]}")


# Criar DataFrame para salvar o resultado
df_bow = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_features
)

# Usar o ID da notícia como identificação
df_bow.insert(
    0,
    "id",
    df["id"]
)

df_bow.to_csv(
    "noticias_ge_bow.csv",
    index=False,
    encoding="utf-8-sig"
)

escrever("Arquivo gerado: noticias_ge_bow.csv")


# ==========================================================
# 3. TF-IDF
# ==========================================================

escrever()
escrever("=" * 50)
escrever("TF-IDF")
escrever("=" * 50)

tfidf_vectorizer = TfidfVectorizer(
    analyzer=lambda tokens: tokens
)

tfidf_matrix = tfidf_vectorizer.fit_transform(
    df["tokens_sem_stopwords"]
)

tfidf_features = tfidf_vectorizer.get_feature_names_out()

escrever(f"Documentos: {tfidf_matrix.shape[0]}")
escrever(f"Palavras no vocabulário: {tfidf_matrix.shape[1]}")
escrever(f"Matriz: {tfidf_matrix.shape[0]} x {tfidf_matrix.shape[1]}")


# Criar DataFrame para salvar o resultado
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_features
)

df_tfidf.insert(
    0,
    "id",
    df["id"]
)

df_tfidf.to_csv(
    "noticias_ge_tfidf.csv",
    index=False,
    encoding="utf-8-sig"
)

escrever("Arquivo gerado: noticias_ge_tfidf.csv")


# ==========================================================
# 4. SIMILARIDADE ENTRE NOTÍCIAS
# ==========================================================

escrever()
escrever("=" * 50)
escrever("SIMILARIDADE")
escrever("=" * 50)

similaridade = cosine_similarity(
    tfidf_matrix
)

escrever(f"Matriz de similaridade: {similaridade.shape[0]} x {similaridade.shape[1]}")


# DataFrame usando os IDs das notícias
df_similaridade = pd.DataFrame(
    similaridade,
    index=df["id"],
    columns=df["id"]
)

df_similaridade.to_csv(
    "noticias_ge_similaridade.csv",
    encoding="utf-8-sig"
)

escrever("Arquivo gerado: noticias_ge_similaridade.csv")


# ==========================================================
# 5. EXEMPLO DE SIMILARIDADE
# ==========================================================

escrever()
escrever("=" * 50)
escrever("EXEMPLO DE SIMILARIDADE")
escrever("=" * 50)

if len(df) >= 2:

    # Pegar as duas primeiras notícias
    id1 = df.iloc[0]["id"]
    id2 = df.iloc[1]["id"]

    valor = similaridade[0][1]

    escrever(f"Notícia 1: {id1}")
    escrever(f"Notícia 2: {id2}")
    escrever(f"Similaridade: {valor:.4f}")


# ==========================================================
# 6. CLUSTERIZAÇÃO (K-MEANS)
# ==========================================================

escrever()
escrever("=" * 50)
escrever("K-MEANS")
escrever("=" * 50)

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

escrever(f"Clusters gerados: {k}")
escrever("Arquivo gerado: noticias_clusters.csv")


# ==========================================================
# 7. PCA 2D
# ==========================================================

escrever()
escrever("=" * 50)
escrever("PCA 2D")
escrever("=" * 50)

pca_2d = PCA(n_components=2)

coordenadas_2d = pca_2d.fit_transform(
    tfidf_matrix.toarray()
)

df_pca_2d = pd.DataFrame({
    "titulo": df["titulo"],
    "cluster": df["cluster"],
    "PCA1": coordenadas_2d[:, 0],
    "PCA2": coordenadas_2d[:, 1]
})

df_pca_2d.to_csv(
    "pca_2d.csv",
    index=False,
    encoding="utf-8-sig"
)

fig = px.scatter(
    df_pca_2d,
    x="PCA1",
    y="PCA2",
    color="cluster",
    hover_name="titulo",
    title="Clusters das notícias - PCA 2D"
)

fig.write_html("pca_2d.html")

escrever("Arquivos gerados:")
escrever("- pca_2d.csv")
escrever("- pca_2d.html")


# ==========================================================
# 8. PCA 3D
# ==========================================================

escrever()
escrever("=" * 50)
escrever("PCA 3D")
escrever("=" * 50)

pca_3d = PCA(n_components=3)

coordenadas_3d = pca_3d.fit_transform(
    tfidf_matrix.toarray()
)

df_pca_3d = pd.DataFrame({
    "titulo": df["titulo"],
    "cluster": df["cluster"],
    "PCA1": coordenadas_3d[:, 0],
    "PCA2": coordenadas_3d[:, 1],
    "PCA3": coordenadas_3d[:, 2]
})

df_pca_3d.to_csv(
    "pca_3d.csv",
    index=False,
    encoding="utf-8-sig"
)

fig = px.scatter_3d(
    df_pca_3d,
    x="PCA1",
    y="PCA2",
    z="PCA3",
    color="cluster",
    hover_name="titulo",
    title="Clusters das notícias - PCA 3D"
)

fig.update_traces(
    marker=dict(size=6)
)

fig.write_html(
    "pca_3d.html"
)

escrever("Arquivos gerados:")
escrever("- pca_3d.csv")
escrever("- pca_3d.html")


# ==========================================================
# 9. PALAVRAS REPRESENTATIVAS DOS CLUSTERS
# ==========================================================

escrever()
escrever("=" * 50)
escrever("PALAVRAS REPRESENTATIVAS")
escrever("=" * 50)

termos = tfidf_vectorizer.get_feature_names_out()

for cluster_id in range(k):

    indices = np.argsort(
        kmeans.cluster_centers_[cluster_id]
    )[::-1]

    principais = termos[
        indices[:10]
    ]

    escrever(
        f"Cluster {cluster_id}: "
        + ", ".join(principais)
    )

# ==========================================================
# FINAL
# ==========================================================

escrever()
escrever("=" * 50)
escrever("PROCESSAMENTO CONCLUÍDO")
escrever("=" * 50)

escrever("Arquivos gerados:")
escrever("- noticias_ge_bow.csv")
escrever("- noticias_ge_tfidf.csv")
escrever("- noticias_ge_similaridade.csv")
escrever("- noticias_ge_tfidf_bow.txt")
escrever("- noticias_clusters.csv")
escrever("- pca_2d.csv")
escrever("- pca_3d.csv")
escrever("- pca_2d.html")
escrever("- pca_3d.html")
relatorio.close()