# python -m pip install pandas numpy scikit-learn matplotlib
# Requer os embeddings gerados antes por word2vec_bert_ge.py.


import ast
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


PASTA_RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_ENTRADA = PASTA_RAIZ / "dados" / "processados" / "noticias_ge_processado.csv"
PASTA_EMBEDDINGS = PASTA_RAIZ / "resultados" / "word2vec_bert"
PASTA_SAIDA = PASTA_RAIZ / "resultados" / "clusterizacao_representacoes"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVO_METRICAS = PASTA_SAIDA / "metricas_por_k.csv"
ARQUIVO_CLUSTERS = PASTA_SAIDA / "clusters_por_representacao.csv"
ARQUIVO_RESUMO = PASTA_SAIDA / "resumo_clusters.csv"
ARQUIVO_RELATORIO = PASTA_SAIDA / "relatorio_clusterizacao_representacoes.txt"
ARQUIVO_GRAFICO_METRICAS = PASTA_SAIDA / "metricas_por_k.png"
ARQUIVO_GRAFICO_PCA = PASTA_SAIDA / "pca_clusters_representacoes.png"
ARQUIVO_GRAFICO_TAMANHOS = PASTA_SAIDA / "tamanho_clusters.png"


RANDOM_STATE = 42

K_MIN = 2
K_MAX = 20

QUANTIDADE_TERMOS = 8
QUANTIDADE_EXEMPLOS = 3

CENTRALIZAR_EMBEDDINGS = True

relatorio = open(
    ARQUIVO_RELATORIO,
    "w",
    encoding="utf-8"
)


def escrever(texto=""):
    relatorio.write(str(texto) + "\n")


df = pd.read_csv(
    ARQUIVO_ENTRADA,
    encoding="utf-8-sig"
)

df["tokens_sem_stopwords"] = df[
    "tokens_sem_stopwords"
].apply(ast.literal_eval)


def carregar_embeddings(nome_arquivo):
    """Lê os embeddings e garante a mesma ordem das notícias."""

    caminho = PASTA_EMBEDDINGS / nome_arquivo

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {caminho}\n"
            "Gere-o antes com word2vec_bert_ge.py."
        )

    embeddings = pd.read_csv(
        caminho,
        encoding="utf-8-sig"
    ).set_index("url")

    colunas = [
        coluna
        for coluna in embeddings.columns
        if coluna.startswith("embedding_")
    ]

    return embeddings.loc[df["url"], colunas].to_numpy()


def preparar_denso(matriz):
    """Centraliza (opcional) e normaliza os embeddings densos."""

    if CENTRALIZAR_EMBEDDINGS:
        matriz = matriz - matriz.mean(axis=0)

    return normalize(matriz)


bow_vectorizer = CountVectorizer(
    analyzer=lambda tokens: tokens
)

matriz_bow = normalize(
    bow_vectorizer.fit_transform(df["tokens_sem_stopwords"])
)

tfidf_vectorizer = TfidfVectorizer(
    analyzer=lambda tokens: tokens
)

matriz_tfidf = tfidf_vectorizer.fit_transform(
    df["tokens_sem_stopwords"]
)

REPRESENTACOES = {
    "BoW": matriz_bow.toarray(),
    "TF-IDF": matriz_tfidf.toarray(),
    "Word2Vec": preparar_denso(
        carregar_embeddings("embeddings_word2vec.csv")
    ),
    "BERT": preparar_denso(
        carregar_embeddings("embeddings_bert.csv")
    ),
}

vocabulario = np.array(tfidf_vectorizer.get_feature_names_out())
matriz_termos = matriz_tfidf.toarray()

escrever("=" * 70)
escrever("CLUSTERIZAÇÃO POR REPRESENTAÇÃO TEXTUAL")
escrever("=" * 70)
escrever(f"Notícias: {len(df)}")
escrever(f"Valores de k testados: {K_MIN} a {K_MAX}")
escrever(f"Embeddings densos centralizados: {CENTRALIZAR_EMBEDDINGS}")
escrever()

for nome, matriz in REPRESENTACOES.items():
    escrever(f"{nome}: {matriz.shape[0]} x {matriz.shape[1]}")


def coesao_por_cluster(matriz, rotulos, centroides):
    """
    Intracluster: cosseno médio entre cada notícia e o centro
    do seu cluster. Quanto maior, mais parecidas as notícias.
    """

    centros = normalize(centroides)
    similaridades = np.sum(matriz * centros[rotulos], axis=1)

    return {
        cluster: similaridades[rotulos == cluster].mean()
        for cluster in np.unique(rotulos)
    }


def separacao_entre_clusters(centroides):
    """
    Intercluster: distância de cosseno média entre os centros.
    Quanto maior, mais diferentes os clusters entre si.
    """

    centros = normalize(centroides)
    similaridades = centros @ centros.T
    fora_diagonal = ~np.eye(len(centros), dtype=bool)

    return (1 - similaridades[fora_diagonal]).mean()


def avaliar(matriz, k):
    """Agrupa com K-Means e calcula todas as métricas."""

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    rotulos = kmeans.fit_predict(matriz)
    coesao = coesao_por_cluster(matriz, rotulos, kmeans.cluster_centers_)

    metricas = {
        "k": k,
        # Intracluster
        "inercia": kmeans.inertia_,
        "coesao_media": np.mean(list(coesao.values())),
        # Intercluster
        "separacao_media": separacao_entre_clusters(kmeans.cluster_centers_),
        # Combinam as duas coisas
        "silhueta": silhouette_score(
            matriz,
            rotulos,
            metric="cosine"
        ),
        "davies_bouldin": davies_bouldin_score(matriz, rotulos),
        "calinski_harabasz": calinski_harabasz_score(matriz, rotulos),
        "menor_cluster": np.bincount(rotulos).min(),
        "maior_cluster": np.bincount(rotulos).max(),
    }

    return metricas, rotulos, kmeans.cluster_centers_


linhas_metricas = []
melhores = {}

for nome, matriz in REPRESENTACOES.items():

    print(f"Avaliando {nome}...")

    for k in range(K_MIN, K_MAX + 1):

        metricas, rotulos, centroides = avaliar(matriz, k)
        metricas["representacao"] = nome
        linhas_metricas.append(metricas)

        if (
            nome not in melhores
            or metricas["silhueta"] > melhores[nome]["silhueta"]
        ):
            melhores[nome] = {
                **metricas,
                "rotulos": rotulos,
                "centroides": centroides,
            }

df_metricas = pd.DataFrame(linhas_metricas)[[
    "representacao", "k", "inercia", "coesao_media",
    "separacao_media", "silhueta", "davies_bouldin",
    "calinski_harabasz", "menor_cluster", "maior_cluster",
]]

df_metricas.to_csv(
    ARQUIVO_METRICAS,
    index=False,
    encoding="utf-8-sig"
)

escrever()
escrever("=" * 70)
escrever("MELHOR k POR REPRESENTAÇÃO (maior silhueta)")
escrever("=" * 70)
escrever(
    "Silhueta: -1 a 1, maior é melhor | Davies-Bouldin: menor é melhor\n"
    "Calinski-Harabasz: maior é melhor | Coesão (intra): maior é melhor\n"
    "Separação (inter): maior é melhor"
)
escrever()

for nome, melhor in melhores.items():
    escrever(
        f"{nome:<9} k={melhor['k']:>2} | "
        f"silhueta={melhor['silhueta']:.4f} | "
        f"DB={melhor['davies_bouldin']:.3f} | "
        f"CH={melhor['calinski_harabasz']:.2f} | "
        f"coesão={melhor['coesao_media']:.3f} | "
        f"separação={melhor['separacao_media']:.3f} | "
        f"tamanhos {melhor['menor_cluster']}-{melhor['maior_cluster']}"
    )

escrever()
escrever("=" * 70)
escrever("MÉTRICAS PARA TODOS OS k")
escrever("=" * 70)
escrever(
    df_metricas.drop(columns=["inercia"]).round(4).to_string(index=False)
)

nomes = list(melhores)

matriz_ari = pd.DataFrame(
    [
        [
            adjusted_rand_score(
                melhores[a]["rotulos"],
                melhores[b]["rotulos"]
            )
            for b in nomes
        ]
        for a in nomes
    ],
    index=nomes,
    columns=nomes
)

escrever()
escrever("=" * 70)
escrever("CONCORDÂNCIA ENTRE REPRESENTAÇÕES (Adjusted Rand Index, melhor k)")
escrever("=" * 70)
escrever(matriz_ari.round(3).to_string())



df_clusters = df[["id", "titulo"]].copy()
linhas_resumo = []

escrever()
escrever("=" * 70)
escrever("CLUSTERS DO MELHOR k DE CADA REPRESENTAÇÃO")
escrever("=" * 70)

for nome, melhor in melhores.items():

    rotulos = melhor["rotulos"]
    df_clusters[f"cluster_{nome}"] = rotulos

    coesao = coesao_por_cluster(
        REPRESENTACOES[nome],
        rotulos,
        melhor["centroides"]
    )

    escrever()
    escrever(f"----- {nome} (k={melhor['k']}) -----")

    for cluster in range(melhor["k"]):

        membros = np.where(rotulos == cluster)[0]

        media_termos = matriz_termos[membros].mean(axis=0)
        termos = vocabulario[
            media_termos.argsort()[::-1][:QUANTIDADE_TERMOS]
        ]

        # Exemplos: as notícias mais próximas do centro do cluster.
        centro = normalize(melhor["centroides"][[cluster]])[0]
        proximidade = REPRESENTACOES[nome][membros] @ centro
        exemplos = df["titulo"].iloc[
            membros[proximidade.argsort()[::-1][:QUANTIDADE_EXEMPLOS]]
        ].tolist()

        linhas_resumo.append({
            "representacao": nome,
            "k": melhor["k"],
            "cluster": cluster,
            "tamanho": len(membros),
            "coesao": round(coesao[cluster], 4),
            "termos": ", ".join(termos),
            "exemplos": " | ".join(exemplos),
        })

        escrever(
            f"Cluster {cluster} ({len(membros)} notícias, "
            f"coesão {coesao[cluster]:.3f}): {', '.join(termos)}"
        )

        for exemplo in exemplos:
            escrever(f"    - {exemplo}")

df_clusters.to_csv(
    ARQUIVO_CLUSTERS,
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(linhas_resumo).to_csv(
    ARQUIVO_RESUMO,
    index=False,
    encoding="utf-8-sig"
)


graficos = [
    ("silhueta", "Silhueta (maior é melhor)"),
    ("davies_bouldin", "Davies-Bouldin (menor é melhor)"),
    ("coesao_media", "Coesão intracluster (maior é melhor)"),
    ("separacao_media", "Separação intercluster (maior é melhor)"),
    ("calinski_harabasz", "Calinski-Harabasz (maior é melhor)"),
    ("inercia_relativa", "Inércia relativa a k=2 (cotovelo)"),
]

df_metricas["inercia_relativa"] = df_metricas["inercia"] / df_metricas.groupby(
    "representacao"
)["inercia"].transform("first")

fig, eixos = plt.subplots(3, 2, figsize=(13, 12))

for eixo, (coluna, titulo) in zip(eixos.ravel(), graficos):

    for nome in REPRESENTACOES:
        dados = df_metricas[df_metricas["representacao"] == nome]
        eixo.plot(dados["k"], dados[coluna], marker="o", markersize=3, label=nome)

    eixo.set_title(titulo)
    eixo.set_xlabel("k (número de clusters)")
    eixo.set_xticks(range(K_MIN, K_MAX + 1, 2))
    eixo.grid(alpha=0.3)

eixos[0, 0].legend()
fig.suptitle("Métricas de clusterização por representação e número de clusters")
fig.tight_layout()
fig.savefig(ARQUIVO_GRAFICO_METRICAS, dpi=150)
plt.close(fig)


fig, eixos = plt.subplots(2, 2, figsize=(13, 11))

for eixo, (nome, melhor) in zip(eixos.ravel(), melhores.items()):

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordenadas = pca.fit_transform(REPRESENTACOES[nome])
    variancia = pca.explained_variance_ratio_.sum() * 100

    pontos = eixo.scatter(
        coordenadas[:, 0],
        coordenadas[:, 1],
        c=melhor["rotulos"],
        cmap="tab20",
        s=12
    )

    eixo.set_title(
        f"{nome} — k={melhor['k']} "
        f"(silhueta {melhor['silhueta']:.3f}; "
        f"PCA explica {variancia:.1f}%)"
    )
    eixo.set_xlabel("Componente principal 1")
    eixo.set_ylabel("Componente principal 2")

fig.suptitle("Clusters do melhor k de cada representação (projeção PCA 2D)")
fig.tight_layout()
fig.savefig(ARQUIVO_GRAFICO_PCA, dpi=150)
plt.close(fig)

fig, eixos = plt.subplots(2, 2, figsize=(13, 9))

for eixo, (nome, melhor) in zip(eixos.ravel(), melhores.items()):

    tamanhos = np.bincount(melhor["rotulos"])

    eixo.bar(range(len(tamanhos)), tamanhos)
    eixo.set_title(f"{nome} — k={melhor['k']}")
    eixo.set_xlabel("Cluster")
    eixo.set_ylabel("Notícias")
    eixo.set_xticks(range(len(tamanhos)))

fig.suptitle("Quantidade de notícias por cluster (melhor k)")
fig.tight_layout()
fig.savefig(ARQUIVO_GRAFICO_TAMANHOS, dpi=150)
plt.close(fig)

escrever()
escrever("=" * 70)
escrever("ARQUIVOS GERADOS")
escrever("=" * 70)

for arquivo in [
    ARQUIVO_METRICAS, ARQUIVO_CLUSTERS, ARQUIVO_RESUMO,
    ARQUIVO_GRAFICO_METRICAS, ARQUIVO_GRAFICO_PCA,
    ARQUIVO_GRAFICO_TAMANHOS, ARQUIVO_RELATORIO,
]:
    escrever(f"- {arquivo.name}")

relatorio.close()

print(f"\nRelatório salvo em: {ARQUIVO_RELATORIO}")
