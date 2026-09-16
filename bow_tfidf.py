# pip install scikit-learn
import pandas as pd
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# CARREGAR DADOS PROCESSADOS
# ==========================================================

df = pd.read_csv(
    "noticias_ge_processado.csv",
    encoding="utf-8-sig"
)

# Converter string para lista
df["tokens_sem_stopwords"] = df["tokens_sem_stopwords"].apply(ast.literal_eval)

# ==========================================================
# BAG OF WORDS
# ==========================================================

print("Gerando BoW...")

bow_vectorizer = CountVectorizer(
    analyzer=lambda tokens: tokens
)

bow_matrix = bow_vectorizer.fit_transform(
    df["tokens_sem_stopwords"]
)

df_bow = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)

df_bow.insert(
    0,
    "link",
    df["link"]
)

df_bow.to_csv(
    "noticias_ge_bow.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# TF-IDF
# ==========================================================

print("Gerando TF-IDF...")

tfidf_vectorizer = TfidfVectorizer(
    analyzer=lambda tokens: tokens
)

tfidf_matrix = tfidf_vectorizer.fit_transform(
    df["tokens_sem_stopwords"]
)

df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

df_tfidf.insert(
    0,
    "link",
    df["link"]
)

df_tfidf.to_csv(
    "noticias_ge_tfidf.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# SIMILARIDADE
# ==========================================================

print("Calculando similaridade...")

similaridade = cosine_similarity(
    tfidf_matrix
)

df_similaridade = pd.DataFrame(
    similaridade,
    index=df["link"],
    columns=df["link"]
)

df_similaridade.to_csv(
    "noticias_ge_similaridade.csv",
    encoding="utf-8-sig"
)

print("Concluído.")