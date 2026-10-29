# python -m pip install pandas numpy scikit-learn matplotlib
# Requer: resultados do Jev (jev_ge.py) e embeddings (word2vec_bert_ge.py).

import ast
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


PASTA_RAIZ = Path(__file__).resolve().parent.parent
PASTA_RESULTADOS = PASTA_RAIZ / "resultados"

ARQUIVO_ENTRADA = PASTA_RAIZ / "dados" / "processados" / "noticias_ge_processado.csv"
ARQUIVO_JEV = PASTA_RESULTADOS / "jev" / "noticias_ge_jev.csv"
PASTA_EMBEDDINGS = PASTA_RESULTADOS / "word2vec_bert"

# Rótulos manuais feitos na base de 100 (classificacao_tfidf_ge.py).
ARQUIVO_GABARITO = (
    PASTA_RESULTADOS / "classificacao_tfidf" / "noticias_ge_rotulacao_tfidf100.csv"
)

PASTA_SAIDA = PASTA_RESULTADOS / "classificacao_representacoes"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVO_METRICAS = PASTA_SAIDA / "metricas_por_tema.csv"
ARQUIVO_RESUMO = PASTA_SAIDA / "resumo_representacoes.csv"
ARQUIVO_PREVISOES_GABARITO = PASTA_SAIDA / "previsoes_gabarito.csv"
ARQUIVO_RELATORIO = PASTA_SAIDA / "relatorio_classificacao_representacoes.txt"
ARQUIVO_GRAFICO_F1 = PASTA_SAIDA / "f1_por_tema.png"
ARQUIVO_GRAFICO_RESUMO = PASTA_SAIDA / "comparacao_representacoes.png"



RANDOM_STATE = 42

PROPORCAO_TESTE = 0.2

TEMAS = [
    "transferencia", "lesao", "resultado_partida", "tecnico",
    "selecao", "competicao", "declaracao", "bastidores",
    "futebol_feminino", "base",
]



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

jev = pd.read_csv(
    ARQUIVO_JEV,
    encoding="utf-8-sig"
).set_index("url")


Y_jev = np.column_stack([
    jev.loc[df["url"], f"jev_tema_{tema}"].str.lower().eq("sim").astype(int)
    for tema in TEMAS
])

gabarito = pd.read_csv(
    ARQUIVO_GABARITO,
    encoding="utf-8-sig"
)

gabarito = gabarito[gabarito["url"].isin(df["url"])]

indices_gabarito = np.array([
    df.index[df["url"] == url][0]
    for url in gabarito["url"]
])

Y_gabarito = np.column_stack([
    gabarito[f"rotulo_{tema}"].str.lower().eq("sim").astype(int)
    for tema in TEMAS
])


indices_restantes = np.setdiff1d(np.arange(len(df)), indices_gabarito)

indices_treino, indices_teste = train_test_split(
    indices_restantes,
    test_size=PROPORCAO_TESTE,
    random_state=RANDOM_STATE
)


def carregar_embeddings(nome_arquivo):
    """Lê os embeddings na mesma ordem das notícias e normaliza."""

    embeddings = pd.read_csv(
        PASTA_EMBEDDINGS / nome_arquivo,
        encoding="utf-8-sig"
    ).set_index("url")

    colunas = [c for c in embeddings.columns if c.startswith("embedding_")]

    return normalize(embeddings.loc[df["url"], colunas].to_numpy())


def representacao_esparsa(vectorizer):
    """
    O vocabulário é aprendido só com as notícias de treino, para
    que o teste e o gabarito não influenciem a representação.
    """

    tokens = df["tokens_sem_stopwords"]

    vectorizer.fit(tokens.iloc[indices_treino])

    return normalize(vectorizer.transform(tokens))


REPRESENTACOES = {
    "BoW": representacao_esparsa(
        CountVectorizer(analyzer=lambda tokens: tokens)
    ),
    "TF-IDF": representacao_esparsa(
        TfidfVectorizer(analyzer=lambda tokens: tokens)
    ),
    "Word2Vec": carregar_embeddings("embeddings_word2vec.csv"),
    "BERT": carregar_embeddings("embeddings_bert.csv"),
}


def metricas(y_real, y_previsto):
    """Métricas da classe positiva ("sim")."""

    return {
        "acuracia": accuracy_score(y_real, y_previsto),
        "precisao": precision_score(y_real, y_previsto, zero_division=0),
        "recall": recall_score(y_real, y_previsto, zero_division=0),
        "f1": f1_score(y_real, y_previsto, zero_division=0),
    }


def criar_modelo(nome):
    if nome == "Maioria (base)":
        return DummyClassifier(strategy="most_frequent")

    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )


linhas_metricas = []
previsoes_teste = {}
previsoes_gabarito = {}

modelos = list(REPRESENTACOES) + ["Maioria (base)"]

for nome in modelos:

    X = REPRESENTACOES.get(nome, REPRESENTACOES["TF-IDF"])

    previsoes_teste[nome] = np.zeros((len(indices_teste), len(TEMAS)), dtype=int)
    previsoes_gabarito[nome] = np.zeros((len(indices_gabarito), len(TEMAS)), dtype=int)

    for posicao, tema in enumerate(TEMAS):

        modelo = criar_modelo(nome)
        modelo.fit(X[indices_treino], Y_jev[indices_treino, posicao])

        previsto_teste = modelo.predict(X[indices_teste])
        previsto_gabarito = modelo.predict(X[indices_gabarito])

        previsoes_teste[nome][:, posicao] = previsto_teste
        previsoes_gabarito[nome][:, posicao] = previsto_gabarito

        linhas_metricas.append({
            "representacao": nome,
            "tema": tema,
            "positivos_treino": int(Y_jev[indices_treino, posicao].sum()),
            "positivos_teste": int(Y_jev[indices_teste, posicao].sum()),
            **metricas(Y_jev[indices_teste, posicao], previsto_teste),
        })

previsoes_gabarito["Jev (LLM)"] = Y_jev[indices_gabarito]

df_metricas = pd.DataFrame(linhas_metricas)

df_metricas.round(4).to_csv(
    ARQUIVO_METRICAS,
    index=False,
    encoding="utf-8-sig"
)

linhas_resumo = []

for nome in list(previsoes_gabarito):

    linha = {"representacao": nome}

    if nome in previsoes_teste:
        dados = df_metricas[df_metricas["representacao"] == nome]
        linha["f1_macro_teste_jev"] = dados["f1"].mean()
        linha["acuracia_media_teste_jev"] = dados["acuracia"].mean()

    linha["acuracia_gabarito"] = accuracy_score(
        Y_gabarito.ravel(),
        previsoes_gabarito[nome].ravel()
    )
    linha["f1_gabarito"] = f1_score(
        Y_gabarito.ravel(),
        previsoes_gabarito[nome].ravel(),
        zero_division=0
    )

    linhas_resumo.append(linha)

df_resumo = pd.DataFrame(linhas_resumo)

df_resumo.round(4).to_csv(
    ARQUIVO_RESUMO,
    index=False,
    encoding="utf-8-sig"
)



def temas_marcados(linha_rotulos):
    return ", ".join(
        tema for tema, valor in zip(TEMAS, linha_rotulos) if valor
    ) or "(nenhum)"


tabela_gabarito = pd.DataFrame({
    "titulo": df["titulo"].iloc[indices_gabarito].to_numpy(),
    "manual": [temas_marcados(l) for l in Y_gabarito],
    **{
        nome: [temas_marcados(l) for l in previsoes]
        for nome, previsoes in previsoes_gabarito.items()
        if nome != "Maioria (base)"
    },
})

tabela_gabarito.to_csv(
    ARQUIVO_PREVISOES_GABARITO,
    index=False,
    encoding="utf-8-sig"
)


escrever("=" * 78)
escrever("CLASSIFICAÇÃO DE TEMAS — RÓTULOS DO JEV x REPRESENTAÇÕES")
escrever("=" * 78)
escrever(f"Notícias: {len(df)}")
escrever(f"Treino (rótulos do Jev): {len(indices_treino)}")
escrever(f"Teste (rótulos do Jev): {len(indices_teste)}")
escrever(f"Gabarito humano (fora do treino): {len(indices_gabarito)} notícias x {len(TEMAS)} temas")
escrever("Classificador: Regressão Logística (class_weight='balanced'), um por tema")
escrever()
escrever("Positivos ('sim') por tema:")
for posicao, tema in enumerate(TEMAS):
    escrever(
        f"  {tema:<18} treino {Y_jev[indices_treino, posicao].sum():>3} | "
        f"teste {Y_jev[indices_teste, posicao].sum():>3} | "
        f"gabarito {Y_gabarito[:, posicao].sum():>2}"
    )

escrever()
escrever("=" * 78)
escrever("RESUMO")
escrever("=" * 78)
escrever(
    "f1_macro_teste_jev: média do F1 dos 10 temas no teste (concordância com o Jev)\n"
    "acuracia/f1_gabarito: acerto em relação aos rótulos humanos (todas as decisões)"
)
escrever()
escrever(df_resumo.round(3).to_string(index=False))

escrever()
escrever("=" * 78)
escrever("F1 NO TESTE POR TEMA (concordância com o Jev)")
escrever("=" * 78)
escrever(
    df_metricas.pivot(index="tema", columns="representacao", values="f1")
    .loc[TEMAS, modelos].round(3).to_string()
)

escrever()
escrever("=" * 78)
escrever("ACURÁCIA NO TESTE POR TEMA (compare com a linha de base)")
escrever("=" * 78)
escrever(
    df_metricas.pivot(index="tema", columns="representacao", values="acuracia")
    .loc[TEMAS, modelos].round(3).to_string()
)

escrever()
escrever("=" * 78)
escrever("GABARITO HUMANO: TEMAS MARCADOS POR NOTÍCIA")
escrever("=" * 78)

for _, linha in tabela_gabarito.iterrows():
    escrever()
    escrever(linha["titulo"])
    for coluna in tabela_gabarito.columns[1:]:
        escrever(f"  {coluna:<10}: {linha[coluna]}")


matriz_f1 = df_metricas.pivot(
    index="tema", columns="representacao", values="f1"
).loc[TEMAS, list(REPRESENTACOES)]

fig, eixo = plt.subplots(figsize=(8, 7))
imagem = eixo.imshow(matriz_f1.to_numpy(), cmap="Blues", vmin=0, vmax=1)

eixo.set_xticks(range(len(matriz_f1.columns)))
eixo.set_xticklabels(matriz_f1.columns)
eixo.set_yticks(range(len(TEMAS)))
eixo.set_yticklabels(TEMAS)

for i in range(len(TEMAS)):
    for j in range(len(matriz_f1.columns)):
        valor = matriz_f1.iloc[i, j]
        eixo.text(
            j, i, f"{valor:.2f}",
            ha="center", va="center",
            color="white" if valor > 0.6 else "black"
        )

eixo.set_title("F1 no teste por tema (concordância com o Jev)")
fig.colorbar(imagem, ax=eixo, fraction=0.046)
fig.tight_layout()
fig.savefig(ARQUIVO_GRAFICO_F1, dpi=150)
plt.close(fig)


resumo_grafico = df_resumo.set_index("representacao")

fig, eixo = plt.subplots(figsize=(10, 5))
posicoes = np.arange(len(resumo_grafico))
largura = 0.38

eixo.bar(
    posicoes - largura / 2,
    resumo_grafico["f1_macro_teste_jev"].fillna(0),
    largura,
    label="F1 macro no teste (rótulos do Jev)"
)
eixo.bar(
    posicoes + largura / 2,
    resumo_grafico["f1_gabarito"],
    largura,
    label="F1 no gabarito humano"
)

eixo.set_xticks(posicoes)
eixo.set_xticklabels(resumo_grafico.index, rotation=15)
eixo.set_ylim(0, 1)
eixo.set_ylabel("F1")
eixo.set_title("Classificação de temas por representação")
eixo.legend()
eixo.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(ARQUIVO_GRAFICO_RESUMO, dpi=150)
plt.close(fig)



escrever()
escrever("=" * 78)
escrever("ARQUIVOS GERADOS")
escrever("=" * 78)

for arquivo in [
    ARQUIVO_METRICAS, ARQUIVO_RESUMO, ARQUIVO_PREVISOES_GABARITO,
    ARQUIVO_GRAFICO_F1, ARQUIVO_GRAFICO_RESUMO, ARQUIVO_RELATORIO,
]:
    escrever(f"- {arquivo.name}")

relatorio.close()

print(f"Relatório salvo em: {ARQUIVO_RELATORIO}")
