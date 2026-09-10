# pip install nltk spacy pandas
# python -m spacy download pt_core_news_sm

import pandas as pd
import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Downloads necessários
for recurso in ["punkt", "punkt_tab", "stopwords", "rslp"]:
    nltk.download(recurso)

# Modelo português
nlp = spacy.load("pt_core_news_sm")

# Stopwords
stopwords_pt = set(stopwords.words("portuguese"))

# Stemmer
stemmer = RSLPStemmer()


def normalizar_tokens(tokens):
    return [
        token.casefold()
        for token in tokens
        if token.isalpha()
    ]


def remover_stopwords(tokens):
    return [
        token
        for token in tokens
        if token not in stopwords_pt
    ]


def lematizar(tokens):
    texto = " ".join(tokens)

    doc = nlp(texto)

    return [
        token.lemma_
        for token in doc
    ]


def aplicar_stemming(tokens):
    return [
        stemmer.stem(token)
        for token in tokens
    ]


def processar_texto(texto):

    if pd.isna(texto):
        texto = ""

    tokens = word_tokenize(
        texto,
        language="portuguese"
    )

    tokens_normalizados = normalizar_tokens(tokens)

    tokens_sem_stopwords = remover_stopwords(
        tokens_normalizados
    )

    lemas = lematizar(tokens_sem_stopwords)

    stems = aplicar_stemming(
        tokens_sem_stopwords
    )

    return pd.Series({
        "tokens": tokens,
        "tokens_normalizados": tokens_normalizados,
        "tokens_sem_stopwords": tokens_sem_stopwords,
        "lemas": lemas,
        "stems": stems
    })


# Lê o CSV da mesma pasta do script
df = pd.read_csv("noticias_ge.csv")

# Processa apenas a coluna texto
colunas_processadas = (
    df["texto"]
    .apply(processar_texto)
)

df_final = pd.concat(
    [df, colunas_processadas],
    axis=1
)

# Salva resultado
df_final.to_csv(
    "noticias_ge_processado.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# GERA O ARQUIVO TXT ORGANIZADO
# ==========================================================

with open(
    "noticias_ge_processado.txt",
    "w",
    encoding="utf-8"
) as arquivo:

    for indice, linha in df_final.iterrows():

        arquivo.write("TEXTO BRUTO:\n")
        arquivo.write(str(linha["texto"]))
        arquivo.write("\n\n")

        arquivo.write("TOKENS NORMALIZADOS:\n")
        arquivo.write(str(linha["tokens_normalizados"]))
        arquivo.write("\n\n")

        arquivo.write("TOKENS SEM STOPWORDS:\n")
        arquivo.write(str(linha["tokens_sem_stopwords"]))
        arquivo.write("\n\n")

        arquivo.write("LEMAS:\n")
        arquivo.write(str(linha["lemas"]))
        arquivo.write("\n\n")

        arquivo.write("STEMS:\n")
        arquivo.write(str(linha["stems"]))
        arquivo.write("\n")

        # Linha em branco para separar os registros
        arquivo.write("\n\n")

print("Arquivo gerado:")
print("noticias_ge_processado.csv")
print("noticias_ge_processado.txt")