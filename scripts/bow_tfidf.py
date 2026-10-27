# pip install scikit-learn pandas plotly
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import ast
import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go


# ==========================================================
# CAMINHOS DO PROJETO
# ==========================================================

# Os scripts ficam em scripts/; a raiz do projeto é a pasta acima.
PASTA_RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_ENTRADA = PASTA_RAIZ / "dados" / "processados" / "noticias_ge_processado.csv"
PASTA_SAIDA = PASTA_RAIZ / "resultados" / "bow_tfidf"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


# ==========================================================
# ARQUIVO DE RELATÓRIO
# ==========================================================

relatorio = open(
    PASTA_SAIDA / "noticias_ge_tfidf_bow.txt",
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
    ARQUIVO_ENTRADA,
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
escrever(
    f"Matriz: {bow_matrix.shape[0]} x {bow_matrix.shape[1]}"
)


# Criar DataFrame para salvar o resultado do BOW
df_bow = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_features
)

df_bow.insert(
    0,
    "id",
    df["id"]
)

df_bow.to_csv(
    PASTA_SAIDA / "noticias_ge_bow.csv",
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
escrever(
    f"Matriz: {tfidf_matrix.shape[0]} x {tfidf_matrix.shape[1]}"
)


# Criar DataFrame para salvar o resultado do TF-IDF
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
    PASTA_SAIDA / "noticias_ge_tfidf.csv",
    index=False,
    encoding="utf-8-sig"
)

escrever("Arquivo gerado: noticias_ge_tfidf.csv")


# ==========================================================
# 4. SIMILARIDADE — BAG OF WORDS
# ==========================================================

escrever()
escrever("=" * 50)
escrever("SIMILARIDADE — BAG OF WORDS")
escrever("=" * 50)

similaridade_bow = cosine_similarity(
    bow_matrix
)

escrever(
    f"Matriz de similaridade BOW: "
    f"{similaridade_bow.shape[0]} x "
    f"{similaridade_bow.shape[1]}"
)


# ==========================================================
# 5. SIMILARIDADE — TF-IDF
# ==========================================================

escrever()
escrever("=" * 50)
escrever("SIMILARIDADE — TF-IDF")
escrever("=" * 50)

similaridade_tfidf = cosine_similarity(
    tfidf_matrix
)

escrever(
    f"Matriz de similaridade TF-IDF: "
    f"{similaridade_tfidf.shape[0]} x "
    f"{similaridade_tfidf.shape[1]}"
)


# ==========================================================
# 6. SALVAR MATRIZ DE SIMILARIDADE TF-IDF
# ==========================================================

df_similaridade_tfidf = pd.DataFrame(
    similaridade_tfidf,
    index=df["id"],
    columns=df["id"]
)

df_similaridade_tfidf.to_csv(
    PASTA_SAIDA / "noticias_ge_similaridade.csv",
    encoding="utf-8-sig"
)

escrever(
    "Arquivo gerado: noticias_ge_similaridade.csv"
)


# ==========================================================
# 7. SALVAR MATRIZ DE SIMILARIDADE BOW
# ==========================================================

df_similaridade_bow = pd.DataFrame(
    similaridade_bow,
    index=df["id"],
    columns=df["id"]
)

df_similaridade_bow.to_csv(
    PASTA_SAIDA / "noticias_ge_similaridade_bow.csv",
    encoding="utf-8-sig"
)

escrever(
    "Arquivo gerado: noticias_ge_similaridade_bow.csv"
)


# ==========================================================
# 8. FUNÇÃO PARA CRIAR OS HEATMAPS
# ==========================================================

# Títulos das notícias
titulos = df["titulo"].astype(str).tolist()

# Textos completos das notícias
descricoes = df["texto"].astype(str).tolist()


# Criar nomes menores para os eixos do gráfico.
# Isso evita que títulos muito grandes deixem
# o gráfico ilegível.
rotulos = []

for i, titulo in enumerate(titulos):

    titulo_curto = titulo

    if len(titulo_curto) > 30:
        titulo_curto = titulo_curto[:30] + "..."

    rotulos.append(
        f"{i + 1} - {titulo_curto}"
    )


def criar_heatmap(matriz, titulo, nome_arquivo):

    # ======================================================
    # DADOS DO TOOLTIP
    # ======================================================

    # Os textos completos NÃO vão em cada célula do heatmap:
    # com 500 notícias são 250.000 células, e repetir dois
    # textos inteiros em cada uma gerava um HTML de ~1,4 GB.
    # Cada texto é gravado uma única vez e exibido num painel
    # abaixo do gráfico quando o mouse passa sobre uma célula.

    painel_js = """
var grafico = document.getElementById('{plot_id}');
var titulos = %s;
var descricoes = %s;

var painel = document.createElement('div');
painel.style.cssText = 'font-family:sans-serif;max-width:1000px;'
    + 'white-space:pre-wrap;padding:8px;';
painel.textContent = 'Passe o mouse sobre uma célula para ver as notícias.';
grafico.parentNode.appendChild(painel);

grafico.on('plotly_hover', function (dados) {
    var ponto = dados.points[0];
    var i = ponto.pointIndex[0];
    var j = ponto.pointIndex[1];

    painel.textContent =
        'Notícia ' + (i + 1) + ': ' + titulos[i] + '\\n\\n'
        + descricoes[i] + '\\n\\n'
        + '----------------------------------------\\n\\n'
        + 'Notícia ' + (j + 1) + ': ' + titulos[j] + '\\n\\n'
        + descricoes[j] + '\\n\\n'
        + 'Similaridade: ' + ponto.z.toFixed(3);
});
""" % (
        # "</" é escapado para um texto não fechar a tag <script>.
        json.dumps(titulos, ensure_ascii=False).replace("</", "<\\/"),
        json.dumps(descricoes, ensure_ascii=False).replace("</", "<\\/")
    )


    # ======================================================
    # CRIAR HEATMAP
    # ======================================================

    fig = go.Figure(

        data=go.Heatmap(

            # 3 casas decimais bastam e reduzem o arquivo.
            z=matriz.round(3),

            x=rotulos,

            y=rotulos,

            colorscale="Blues",

            zmin=0,

            zmax=1,

            # Os rótulos dos eixos já trazem número e título curto.
            hovertemplate=

                "<b>%{y}</b><br>"

                "<b>%{x}</b><br><br>"

                "<b>Similaridade:</b> "
                "%{z:.3f}"

                "<extra></extra>"
        )
    )


    # ======================================================
    # CONFIGURAÇÃO DO GRÁFICO
    # ======================================================

    fig.update_layout(

        title=titulo,

        xaxis_title="Notícia",

        yaxis_title="Notícia",

        width=1000,

        height=900,

        xaxis=dict(
            tickangle=-60
        ),

        yaxis=dict(
            autorange="reversed"
        )
    )


    # ======================================================
    # SALVAR HTML INTERATIVO
    # ======================================================

    pio.write_html(
        fig,
        file=nome_arquivo,
        auto_open=False,
        include_plotlyjs=True,
        post_script=painel_js
    )


    # ======================================================
    # MOSTRAR NA TELA
    # ======================================================

    fig.show()

    print(
        f"Gráfico interativo salvo em: {nome_arquivo}"
    )


# ==========================================================
# 9. GERAR HEATMAP — BAG OF WORDS
# ==========================================================

criar_heatmap(

    similaridade_bow,

    "Similaridade entre notícias — Bag of Words",

    PASTA_SAIDA / "similaridade_bow.html"

)


# ==========================================================
# 10. GERAR HEATMAP — TF-IDF
# ==========================================================

criar_heatmap(

    similaridade_tfidf,

    "Similaridade entre notícias — TF-IDF",

    PASTA_SAIDA / "similaridade_tfidf.html"

)


# ==========================================================
# 11. EXEMPLO DE SIMILARIDADE
# ==========================================================

escrever()
escrever("=" * 50)
escrever("EXEMPLO DE SIMILARIDADE")
escrever("=" * 50)

if len(df) >= 2:

    id1 = df.iloc[0]["id"]
    id2 = df.iloc[1]["id"]

    valor_bow = similaridade_bow[0][1]

    valor_tfidf = similaridade_tfidf[0][1]

    escrever(f"Notícia 1: {id1}")
    escrever(f"Notícia 2: {id2}")

    escrever(
        f"Similaridade BOW: {valor_bow:.4f}"
    )

    escrever(
        f"Similaridade TF-IDF: {valor_tfidf:.4f}"
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
escrever("- noticias_ge_similaridade_bow.csv")
escrever("- noticias_ge_tfidf_bow.txt")


relatorio.close()