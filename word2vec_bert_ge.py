# python -m pip install gensim

import ast
import re
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModel

nltk.download("stopwords", quiet=True)

# ==========================================================
# CARREGAR DADOS DO PROJETO
# ==========================================================

df = pd.read_csv(
    "noticias_ge_processado.csv",
    encoding="utf-8-sig"
)

df["tokens_sem_stopwords"] = df["tokens_sem_stopwords"].apply(ast.literal_eval)

stopwords_pt = set(stopwords.words("portuguese"))

# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def tokenizar(texto):
    return texto.split()

def normalizar(tokens):
    retorno = []

    for token in tokens:
        token = token.lower()
        token = re.sub(r"[^\wÀ-ÿ-]", "", token)

        if token:
            retorno.append(token)

    return retorno

def remover_stopwords(tokens):
    return [
        token
        for token in tokens
        if token not in stopwords_pt
    ]

# ==========================================================
# WORD2VEC
# ==========================================================

modelo_w2v = Word2Vec(
    sentences=df["tokens_sem_stopwords"].tolist(),
    vector_size=190,
    window=5,
    min_count=1,
    sg=1,
    seed=42,
    workers=1,
    epochs=200
)

def vetor_documento_word2vec(tokens):

    vetores = [
        modelo_w2v.wv[token]
        for token in tokens
        if token in modelo_w2v.wv
    ]

    if not vetores:
        return np.zeros(modelo_w2v.vector_size)

    return np.mean(vetores, axis=0)

embeddings_word2vec = np.vstack(
    df["tokens_sem_stopwords"].apply(
        vetor_documento_word2vec
    )
)

# ==========================================================
# BERTIMBAU
# ==========================================================

MODELO_BERT = "neuralmind/bert-base-portuguese-cased"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer_bert = AutoTokenizer.from_pretrained(MODELO_BERT)

modelo_bert = AutoModel.from_pretrained(MODELO_BERT)
modelo_bert.to(device)
modelo_bert.eval()

def mean_pooling(last_hidden_state, attention_mask):

    mascara = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    soma = torch.sum(
        last_hidden_state * mascara,
        dim=1
    )

    quantidade = torch.clamp(
        mascara.sum(dim=1),
        min=1e-9
    )

    return soma / quantidade

def gerar_embeddings_bert(textos):

    todos = []

    for texto in textos:

        entradas = tokenizer_bert(
            texto,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        entradas = {
            k: v.to(device)
            for k, v in entradas.items()
        }

        with torch.no_grad():
            saida = modelo_bert(**entradas)

        emb = mean_pooling(
            saida.last_hidden_state,
            entradas["attention_mask"]
        )

        todos.append(
            emb.cpu().numpy()[0]
        )

    return np.array(todos)

embeddings_bert = gerar_embeddings_bert(
    df["texto"].tolist()
)

# ==========================================================
# BUSCA
# ==========================================================

def buscar_top_k(vetor_consulta, matriz_documentos, k=5):

    similaridades = cosine_similarity(
        vetor_consulta.reshape(1, -1),
        matriz_documentos
    )[0]

    indices = np.argsort(similaridades)[::-1][:k]

    return pd.DataFrame({
        "titulo": df.iloc[indices]["titulo"].values,
        "data": df.iloc[indices]["data_publicacao"].values,
        "url": df.iloc[indices]["url"].values,
        "similaridade": similaridades[indices]
    })

def representar_consulta_word2vec(texto):

    tokens = tokenizar(texto)
    tokens = normalizar(tokens)
    tokens = remover_stopwords(tokens)

    conhecidos = [
        t for t in tokens
        if t in modelo_w2v.wv
    ]

    if not conhecidos:
        return None

    return np.mean(
        [modelo_w2v.wv[t] for t in conhecidos],
        axis=0
    )

def representar_consulta_bert(texto):

    return gerar_embeddings_bert([texto])[0]

# ==========================================================
# EXECUÇÃO
# ==========================================================

consulta = input("Consulta: ").strip()

print("\n========== WORD2VEC ==========")

vetor_w2v = representar_consulta_word2vec(
    consulta
)

if vetor_w2v is not None:
    print(
        buscar_top_k(
            vetor_w2v,
            embeddings_word2vec,
            5
        )
    )
else:
    print("Nenhum termo encontrado no vocabulário.")

print("\n========== BERTIMBAU ==========")

vetor_bert = representar_consulta_bert(
    consulta
)

print(
    buscar_top_k(
        vetor_bert,
        embeddings_bert,
        5
    )
)
