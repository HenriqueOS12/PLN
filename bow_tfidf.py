# pip install scikit-learn
import pandas as pd
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# 1. CARREGAR DADOS PROCESSADOS
# ==========================================================

print("=" * 50)
print("CARREGANDO DADOS")
print("=" * 50)

df = pd.read_csv(
    "noticias_ge_processado.csv",
    encoding="utf-8-sig"
)

# O CSV salva a lista de tokens como texto.
# Aqui transformamos novamente em lista Python.
df["tokens_sem_stopwords"] = df[
    "tokens_sem_stopwords"
].apply(ast.literal_eval)

print(f"Documentos encontrados: {len(df)}")


# ==========================================================
# 2. BAG OF WORDS
# ==========================================================

print()
print("=" * 50)
print("BAG OF WORDS")
print("=" * 50)

bow_vectorizer = CountVectorizer(
    analyzer=lambda tokens: tokens
)

bow_matrix = bow_vectorizer.fit_transform(
    df["tokens_sem_stopwords"]
)

bow_features = bow_vectorizer.get_feature_names_out()

print(f"Documentos: {bow_matrix.shape[0]}")
print(f"Palavras no vocabulário: {bow_matrix.shape[1]}")
print(f"Matriz: {bow_matrix.shape[0]} x {bow_matrix.shape[1]}")


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

print("Arquivo gerado: noticias_ge_bow.csv")


# ==========================================================
# 3. TF-IDF
# ==========================================================

print()
print("=" * 50)
print("TF-IDF")
print("=" * 50)

tfidf_vectorizer = TfidfVectorizer(
    analyzer=lambda tokens: tokens
)

tfidf_matrix = tfidf_vectorizer.fit_transform(
    df["tokens_sem_stopwords"]
)

tfidf_features = tfidf_vectorizer.get_feature_names_out()

print(f"Documentos: {tfidf_matrix.shape[0]}")
print(f"Palavras no vocabulário: {tfidf_matrix.shape[1]}")
print(f"Matriz: {tfidf_matrix.shape[0]} x {tfidf_matrix.shape[1]}")


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

print("Arquivo gerado: noticias_ge_tfidf.csv")


# ==========================================================
# 4. SIMILARIDADE ENTRE NOTÍCIAS
# ==========================================================

print()
print("=" * 50)
print("SIMILARIDADE")
print("=" * 50)

similaridade = cosine_similarity(
    tfidf_matrix
)

print(f"Matriz de similaridade: {similaridade.shape[0]} x {similaridade.shape[1]}")


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

print("Arquivo gerado: noticias_ge_similaridade.csv")


# ==========================================================
# 5. EXEMPLO DE SIMILARIDADE
# ==========================================================

print()
print("=" * 50)
print("EXEMPLO DE SIMILARIDADE")
print("=" * 50)

if len(df) >= 2:

    # Pegar as duas primeiras notícias
    id1 = df.iloc[0]["id"]
    id2 = df.iloc[1]["id"]

    valor = similaridade[0][1]

    print(f"Notícia 1: {id1}")
    print(f"Notícia 2: {id2}")
    print(f"Similaridade: {valor:.4f}")


# ==========================================================
# FINAL
# ==========================================================

print()
print("=" * 50)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 50)

print("Arquivos gerados:")
print("- noticias_ge_bow.csv")
print("- noticias_ge_tfidf.csv")
print("- noticias_ge_similaridade.csv")