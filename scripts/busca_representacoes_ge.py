# python -m pip install pandas numpy scikit-learn gensim torch transformers nltk matplotlib
# Requer os embeddings BERT gerados antes por word2vec_bert_ge.py.

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer


PASTA_RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_ENTRADA = PASTA_RAIZ / "dados" / "processados" / "noticias_ge_processado.csv"
PASTA_EMBEDDINGS = PASTA_RAIZ / "resultados" / "word2vec_bert"
PASTA_SAIDA = PASTA_RAIZ / "resultados" / "busca_representacoes"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVO_RESULTADOS = PASTA_SAIDA / "resultados_busca.csv"
ARQUIVO_PRECISAO = PASTA_SAIDA / "precisao_por_representacao.csv"
ARQUIVO_RELATORIO = PASTA_SAIDA / "relatorio_busca_representacoes.txt"
ARQUIVO_GRAFICO = PASTA_SAIDA / "precisao_busca.png"


RANDOM_STATE = 42
QUANTIDADE_RESULTADOS = 5

TAMANHO_VETOR_W2V = 100
MODELO_BERT = "neuralmind/bert-base-portuguese-cased"
TAMANHO_MAXIMO_BERT = 256

PESO_TFIDF_HIBRIDO = 0.5

CONSULTAS = [
    ("Corinthians", r"corinthians|timão"),
    ("Cristiano Ronaldo", r"cristiano|cr7"),
    ("Ancelotti convocação da seleção", r"seleç|convoca|ancelotti"),
    ("lesão de jogador", r"les[ãa]o|les[õo]es|lesionad|machuc|cirurgia|desfalc|recupera|\bDM\b"),
    ("contratação de atacante", r"contrata|reforç|acerta|negocia|proposta|anuncia|investida|transfer|janela"),
    ("árbitro polêmica pênalti", r"árbitr|arbitragem|pênalti|var\b|expuls"),
    ("onde assistir ao vivo", r"onde assistir|ao vivo"),
    ("time perde clássico e torcida protesta", r"protest|vaia|torcida|torcedor"),
    ("Corinthian", r"corinthians|timão"),
    ("Bola de Ouro", r"bola de ouro"),
]

stopwords_pt = set(stopwords.words("portuguese"))


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
).fillna("")

df["tokens_sem_stopwords"] = df[
    "tokens_sem_stopwords"
].apply(ast.literal_eval)


def tokens_bow(consulta):
    """Mesmo tratamento de processamento.py: tokeniza, mantém só
    palavras, converte para minúsculas e remove stopwords."""

    tokens = word_tokenize(consulta, language="portuguese")

    return [
        token.casefold()
        for token in tokens
        if token.isalpha() and token.casefold() not in stopwords_pt
    ]


def tokens_word2vec(consulta):
    """Mesmo tratamento da consulta em word2vec_bert_ge.py."""

    texto = consulta.casefold()
    texto = re.sub(r"[^a-zA-ZÀ-ÿ0-9]+", " ", texto)

    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", texto)


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


def buscar_esparso(vectorizer, matriz, consulta):
    """Similaridade de cosseno entre a consulta e as notícias."""

    tokens = tokens_bow(consulta)
    vetor = normalize(vectorizer.transform([tokens]))

    # Consulta sem nenhum termo do vocabulário: sem resultado.
    if vetor.nnz == 0:
        return None

    return (matriz @ vetor.T).toarray().ravel()


print("Treinando Word2Vec...")

modelo_w2v = Word2Vec(
    sentences=[t for t in df["tokens_sem_stopwords"] if t],
    vector_size=TAMANHO_VETOR_W2V,
    window=5,
    min_count=1,
    sg=1,
    seed=RANDOM_STATE,
    workers=1,
    epochs=100
)


def media_word2vec(tokens):
    vetores = [modelo_w2v.wv[t] for t in tokens if t in modelo_w2v.wv]

    if not vetores:
        return np.zeros(TAMANHO_VETOR_W2V, dtype=np.float32)

    return np.mean(vetores, axis=0)


embeddings_w2v = np.asarray(
    [media_word2vec(tokens) for tokens in df["tokens_sem_stopwords"]],
    dtype=np.float32
)

salvos_w2v = pd.read_csv(
    PASTA_EMBEDDINGS / "embeddings_word2vec.csv",
    encoding="utf-8-sig"
).set_index("url")

salvos_w2v = salvos_w2v.loc[
    df["url"],
    [c for c in salvos_w2v.columns if c.startswith("embedding_")]
].to_numpy()

w2v_igual_ao_salvo = np.allclose(embeddings_w2v, salvos_w2v, atol=1e-4)

embeddings_w2v = normalize(embeddings_w2v)


def buscar_word2vec(consulta):
    vetor = media_word2vec(tokens_word2vec(consulta))

    if not vetor.any():
        return None

    return embeddings_w2v @ normalize([vetor])[0]


print("Carregando BERTimbau...")

tokenizador = AutoTokenizer.from_pretrained(MODELO_BERT)
modelo_bert = AutoModel.from_pretrained(MODELO_BERT)
modelo_bert.eval()

salvos_bert = pd.read_csv(
    PASTA_EMBEDDINGS / "embeddings_bert.csv",
    encoding="utf-8-sig"
).set_index("url")

embeddings_bert = salvos_bert.loc[
    df["url"],
    [c for c in salvos_bert.columns if c.startswith("embedding_")]
].to_numpy()


media_bert = embeddings_bert.mean(axis=0)
embeddings_bert_centralizados = normalize(embeddings_bert - media_bert)


def vetor_bert(consulta):
    """Mesma média de tokens de word2vec_bert_ge.py (sem [CLS],
    [SEP] e padding), normalizada."""

    entradas = tokenizador(
        [consulta],
        truncation=True,
        max_length=TAMANHO_MAXIMO_BERT,
        return_tensors="pt"
    )

    with torch.no_grad():
        vetores = modelo_bert(**entradas).last_hidden_state[0]

    especiais = torch.isin(
        entradas["input_ids"][0],
        torch.tensor(tokenizador.all_special_ids)
    )

    vetor = vetores[~especiais].mean(dim=0).numpy()

    return vetor / np.linalg.norm(vetor)


def buscar_bert(consulta):
    return embeddings_bert @ vetor_bert(consulta)


def buscar_bert_centralizado(consulta):
    vetor = normalize([vetor_bert(consulta) - media_bert])[0]

    return embeddings_bert_centralizados @ vetor


def escala_0_1(valores):
    """Coloca as notas de uma consulta na faixa de 0 a 1."""

    minimo, maximo = valores.min(), valores.max()

    if maximo == minimo:
        return np.zeros_like(valores)

    return (valores - minimo) / (maximo - minimo)


def buscar_hibrido(consulta):
    """Soma ponderada do TF-IDF (palavras exatas) com o Word2Vec
    (palavras parecidas). Se um dos dois não reconhece a consulta,
    usa só o outro."""

    notas_tfidf = buscar_esparso(tfidf_vectorizer, matriz_tfidf, consulta)
    notas_w2v = buscar_word2vec(consulta)

    if notas_tfidf is None:
        return notas_w2v
    if notas_w2v is None:
        return notas_tfidf

    return (
        PESO_TFIDF_HIBRIDO * escala_0_1(notas_tfidf)
        + (1 - PESO_TFIDF_HIBRIDO) * escala_0_1(notas_w2v)
    )


REPRESENTACOES = {
    "BoW": lambda c: buscar_esparso(bow_vectorizer, matriz_bow, c),
    "TF-IDF": lambda c: buscar_esparso(tfidf_vectorizer, matriz_tfidf, c),
    "Word2Vec": buscar_word2vec,
    "BERT": buscar_bert,
    "BERT centralizado": buscar_bert_centralizado,
    "Híbrido TF-IDF+W2V": buscar_hibrido,
}

escrever("=" * 78)
escrever("BUSCA POR REPRESENTAÇÃO TEXTUAL")
escrever("=" * 78)
escrever(f"Notícias: {len(df)}")
escrever(f"Consultas: {len(CONSULTAS)}")
escrever(f"Word2Vec idêntico ao de word2vec_bert_ge.py: {w2v_igual_ao_salvo}")
escrever(
    "[R] = relevante pela heurística (padrão da consulta aparece no título)"
)

linhas = []

for consulta, padrao in CONSULTAS:

    print(f"Consulta: {consulta}")

    escrever()
    escrever("=" * 78)
    escrever(f"CONSULTA: {consulta}")
    escrever(f"Tokens BoW/TF-IDF: {tokens_bow(consulta)}")
    escrever(
        "Tokens Word2Vec conhecidos: "
        f"{[t for t in tokens_word2vec(consulta) if t in modelo_w2v.wv]}"
    )
    escrever("=" * 78)

    for nome, buscar in REPRESENTACOES.items():

        notas = buscar(consulta)

        escrever()
        escrever(f"--- {nome} ---")

        if notas is None:
            escrever("Nenhum resultado: nenhum termo da consulta é conhecido.")
            continue

        melhores = np.argsort(notas)[::-1][:QUANTIDADE_RESULTADOS]

        for posicao, indice in enumerate(melhores, start=1):

            titulo = df["titulo"].iloc[indice]
            relevante = bool(re.search(padrao, titulo, re.IGNORECASE))

            linhas.append({
                "consulta": consulta,
                "representacao": nome,
                "posicao": posicao,
                "similaridade": round(float(notas[indice]), 4),
                "relevante_titulo": relevante,
                "titulo": titulo,
                "url": df["url"].iloc[indice],
            })

            escrever(
                f"{posicao}. {'[R]' if relevante else '[ ]'} "
                f"{notas[indice]:.3f} | {titulo}"
            )

df_resultados = pd.DataFrame(linhas)

df_resultados.to_csv(
    ARQUIVO_RESULTADOS,
    index=False,
    encoding="utf-8-sig"
)

linhas_precisao = []

for nome in REPRESENTACOES:

    for consulta, _ in CONSULTAS:

        dados = df_resultados[
            (df_resultados["representacao"] == nome)
            & (df_resultados["consulta"] == consulta)
        ]

        linhas_precisao.append({
            "representacao": nome,
            "consulta": consulta,
            "precisao_top3": dados[dados["posicao"] <= 3]["relevante_titulo"].sum() / 3,
            "precisao_top5": dados["relevante_titulo"].sum() / 5,
        })

df_precisao = pd.DataFrame(linhas_precisao)

df_precisao.to_csv(
    ARQUIVO_PRECISAO,
    index=False,
    encoding="utf-8-sig"
)

media_precisao = df_precisao.groupby(
    "representacao",
    sort=False
)[["precisao_top3", "precisao_top5"]].mean()

escrever()
escrever("=" * 78)
escrever("PRECISÃO MÉDIA (heurística pelo título)")
escrever("=" * 78)
escrever(media_precisao.round(3).to_string())

escrever()
escrever("PRECISÃO NO TOP-5 POR CONSULTA")
escrever(
    df_precisao.pivot(
        index="consulta",
        columns="representacao",
        values="precisao_top5"
    )[list(REPRESENTACOES)].round(2).to_string()
)

fig, eixo = plt.subplots(figsize=(11, 5))

posicoes = np.arange(len(media_precisao))
largura = 0.38

eixo.bar(posicoes - largura / 2, media_precisao["precisao_top3"], largura, label="Top-3")
eixo.bar(posicoes + largura / 2, media_precisao["precisao_top5"], largura, label="Top-5")

eixo.set_xticks(posicoes)
eixo.set_xticklabels(media_precisao.index, rotation=15)
eixo.set_ylim(0, 1)
eixo.set_ylabel("Precisão média")
eixo.set_title(
    f"Precisão da busca em {len(CONSULTAS)} consultas "
    "(relevância aproximada pelo título)"
)
eixo.legend()
eixo.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(ARQUIVO_GRAFICO, dpi=150)
plt.close(fig)

escrever()
escrever("=" * 78)
escrever("ARQUIVOS GERADOS")
escrever("=" * 78)

for arquivo in [ARQUIVO_RESULTADOS, ARQUIVO_PRECISAO, ARQUIVO_GRAFICO, ARQUIVO_RELATORIO]:
    escrever(f"- {arquivo.name}")

relatorio.close()

print(f"\nRelatório salvo em: {ARQUIVO_RELATORIO}")
