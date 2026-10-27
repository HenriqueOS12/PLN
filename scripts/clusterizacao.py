import pandas as pd
import ast
import json
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# CAMINHOS DO PROJETO
# ==========================================================

# Os scripts ficam em scripts/; a raiz do projeto é a pasta acima.
PASTA_RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_ENTRADA = PASTA_RAIZ / "dados" / "processados" / "noticias_ge_processado.csv"
PASTA_SAIDA = PASTA_RAIZ / "resultados" / "clusterizacao"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


# ==========================================================
# CARREGAR DADOS
# ==========================================================

df = pd.read_csv(
    ARQUIVO_ENTRADA,
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
    PASTA_SAIDA / "noticias_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# PCA 2D E 3D DOS CLUSTERS
# ==========================================================

from sklearn.decomposition import PCA
import plotly.express as px


# ==========================================================
# PAINEL COM O TEXTO COMPLETO DA NOTÍCIA
# ==========================================================

# O texto completo não cabe no tooltip (fica numa linha só
# e é cortado na tela). Ele é exibido num painel abaixo do
# gráfico quando o mouse passa sobre um ponto.
# Cada ponto guarda em customdata a posição da notícia no df.

def criar_painel_js():

    noticias = [
        {
            "titulo": str(linha["titulo"]),
            "id": str(linha["id"]),
            "cluster": str(linha["cluster"]),
            "texto": str(linha["texto"])
        }
        for _, linha in df.iterrows()
    ]

    # "</" é escapado para um texto não fechar a tag <script>.
    dados_json = json.dumps(
        noticias,
        ensure_ascii=False
    ).replace("</", "<\\/")

    return """
var grafico = document.getElementById('{plot_id}');
var noticias = %s;

var painel = document.createElement('div');
painel.style.cssText = 'font-family:sans-serif;max-width:1200px;'
    + 'white-space:pre-wrap;padding:8px;';
painel.textContent = 'Passe o mouse sobre um ponto para ler a notícia completa.';
grafico.parentNode.appendChild(painel);

grafico.on('plotly_hover', function (dados) {
    var n = noticias[dados.points[0].customdata[0]];

    painel.textContent =
        n.titulo + '\\n'
        + 'Cluster: ' + n.cluster + '\\n'
        + n.id + '\\n\\n'
        + n.texto;
});
""" % dados_json


painel_js = criar_painel_js()


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
    PASTA_SAIDA / "pca_2d.csv",
    index=False,
    encoding="utf-8-sig"
)

# Posição de cada notícia, usada pelo painel do texto completo.
df_pca_2d["n"] = range(len(df))


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

        "cluster": True,

        "PCA1": False,

        "PCA2": False

    },

    custom_data=["n"],

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

    PASTA_SAIDA / "clusters_pca_2d.html",

    auto_open=False,

    post_script=painel_js

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

    PASTA_SAIDA / "pca_3d.csv",

    index=False,

    encoding="utf-8-sig"

)

# Posição de cada notícia, usada pelo painel do texto completo.
df_pca_3d["n"] = range(len(df))


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

        "cluster": True,

        "PCA1": False,

        "PCA2": False,

        "PCA3": False

    },

    custom_data=["n"],

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

    PASTA_SAIDA / "clusters_pca_3d.html",

    auto_open=False,

    post_script=painel_js

)


# Mostrar gráfico
fig_3d.show()   