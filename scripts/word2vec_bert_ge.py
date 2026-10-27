
import ast
import os
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIGURACOES
# ============================================================

# Os scripts ficam em scripts/; a raiz do projeto e a pasta acima.
PASTA_RAIZ = Path(__file__).resolve().parent.parent
PASTA_SAIDA = PASTA_RAIZ / "resultados" / "word2vec_bert"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVO_CSV = PASTA_RAIZ / "dados" / "processados" / "noticias_ge_processado.csv"

ARQUIVO_EMBEDDINGS_W2V = PASTA_SAIDA / "embeddings_word2vec.csv"
ARQUIVO_EMBEDDINGS_BERT = PASTA_SAIDA / "embeddings_bert.csv"
ARQUIVO_RELATORIO = PASTA_SAIDA / "relatorio_word2vec_bert.txt"

MODELO_BERT = "neuralmind/bert-base-portuguese-cased"

QUANTIDADE_RESULTADOS = 5
TAMANHO_VETOR_W2V = 100
TAMANHO_LOTE_BERT = 8
TAMANHO_MAXIMO_BERT = 256


# ============================================================
# FUNCOES AUXILIARES
# ============================================================

def texto_seguro(valor):
    """Converte valores do CSV em texto."""
    if valor is None or pd.isna(valor):
        return ""

    return str(valor)


def normalizar_texto(texto):
    """Normaliza o texto sem remover acentos das palavras."""
    texto = texto_seguro(texto).casefold()
    texto = re.sub(r"https?://\S+|www\\.\S+", " ", texto)
    texto = re.sub(r"[^a-zA-ZÀ-ÿ0-9]+", " ", texto)

    return texto.strip()


def tokenizar(texto):
    """Extrai palavras inteiras, preservando os acentos."""
    texto = normalizar_texto(texto)

    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", texto)

def ler_tokens_processados(valor):
    """Converte uma lista armazenada como texto no CSV em lista Python."""

    if isinstance(valor, list):
        return [str(t).casefold() for t in valor if str(t).strip()]

    if not isinstance(valor, str) or not valor.strip():
        return []

    try:
        resultado = ast.literal_eval(valor)

        if isinstance(resultado, (list, tuple)):
            return [
                str(t).casefold()
                for t in resultado
                if str(t).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    # Caso a coluna nao contenha uma lista Python valida.
    return tokenizar(valor)

def similaridade_cosseno(vetor_a, vetor_b):
    """Calcula a similaridade de cosseno entre dois vetores."""

    norma_a = np.linalg.norm(vetor_a)
    norma_b = np.linalg.norm(vetor_b)

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return float(np.dot(vetor_a, vetor_b) / (norma_a * norma_b))

def criar_texto_documento(linha):
    """Monta o texto usado na representacao de cada noticia."""

    titulo = texto_seguro(linha.get("titulo", ""))
    texto = texto_seguro(linha.get("texto", ""))

    return (titulo + " " + texto).strip()


# ============================================================
# CARREGAR NOTICIAS
# ============================================================

if not os.path.exists(ARQUIVO_CSV):
    raise FileNotFoundError(
        f"Arquivo nao encontrado: {ARQUIVO_CSV}\n"
        "Gere o CSV com processamento.py (dados/processados)."
    )

df = pd.read_csv(ARQUIVO_CSV, encoding="utf-8-sig")

colunas_obrigatorias = [
    "titulo",
    "data_publicacao",
    "url"
]

for coluna in colunas_obrigatorias:
    if coluna not in df.columns:
        raise ValueError(
            f"A coluna obrigatoria '{coluna}' nao existe no CSV."
        )

if "texto" not in df.columns:
    df["texto"] = ""

if "tokens_sem_stopwords" not in df.columns:
    df["tokens_sem_stopwords"] = ""

df = df.fillna("")

# Textos originais usados na busca.
df["_texto_busca"] = df.apply(criar_texto_documento, axis=1)

# Tokens processados, se existirem.
df["_tokens_processados"] = df[
    "tokens_sem_stopwords"
].apply(ler_tokens_processados)

# Tokens extraidos dos textos originais.
df["_tokens_originais"] = df["_texto_busca"].apply(tokenizar)

if len(df) == 0:
    raise ValueError("O CSV nao possui noticias.")

print(f"Noticias carregadas: {len(df)}")


# ============================================================
# TREINAR WORD2VEC
# ============================================================

print("\nTreinando Word2Vec...")

sentencas = [
    tokens
    for tokens in df["_tokens_processados"].tolist()
    if tokens
]

if not sentencas:
    raise ValueError(
        "Nao foi possivel treinar Word2Vec: a coluna "
        "'tokens_sem_stopwords' nao possui tokens validos."
    )

modelo_w2v = Word2Vec(
    sentences=sentencas,
    vector_size=TAMANHO_VETOR_W2V,
    window=5,
    min_count=1,
    sg=1,
    seed=42,
    workers=1,
    epochs=100
)

print("Vocabulario Word2Vec:", len(modelo_w2v.wv))

# ============================================================
# CRIAR EMBEDDINGS WORD2VEC DAS NOTICIAS
# ============================================================

print("\nGerando embeddings Word2Vec...")

embeddings_w2v = []

for tokens in df["_tokens_processados"]:

    vetores = [
        modelo_w2v.wv[token]
        for token in tokens
        if token in modelo_w2v.wv
    ]

    if vetores:
        vetor_documento = np.mean(vetores, axis=0)
    else:
        vetor_documento = np.zeros(
            TAMANHO_VETOR_W2V,
            dtype=np.float32
        )

    embeddings_w2v.append(vetor_documento)

embeddings_w2v = np.asarray(
    embeddings_w2v,
    dtype=np.float32
)

dados_w2v = df[
    ["titulo", "data_publicacao", "url"]
].copy()

for i in range(TAMANHO_VETOR_W2V):
    dados_w2v[f"embedding_{i}"] = embeddings_w2v[:, i]

dados_w2v.to_csv(
    ARQUIVO_EMBEDDINGS_W2V,
    index=False,
    encoding="utf-8-sig"
)

print("Arquivo salvo:", ARQUIVO_EMBEDDINGS_W2V)

# ============================================================
# CARREGAR BERTIMBAU
# ============================================================

print("\nCarregando BERTimbau...")
print("Na primeira execucao, o modelo pode precisar ser baixado.")

dispositivo = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo:", dispositivo)

tokenizador = AutoTokenizer.from_pretrained(MODELO_BERT)

modelo_bert = AutoModel.from_pretrained(MODELO_BERT)
modelo_bert.to(dispositivo)
modelo_bert.eval()


# ============================================================
# GERAR EMBEDDINGS BERT
# ============================================================

def gerar_embeddings_bert(lista_textos):
    """Gera embeddings usando a media dos tokens do BERTimbau."""

    todos_embeddings = []

    for inicio in range(0, len(lista_textos), TAMANHO_LOTE_BERT):

        lote = lista_textos[
            inicio:inicio + TAMANHO_LOTE_BERT
        ]

        entradas = tokenizador(
            lote,
            padding=True,
            truncation=True,
            max_length=TAMANHO_MAXIMO_BERT,
            return_tensors="pt"
        )

        entradas = {
            chave: valor.to(dispositivo)
            for chave, valor in entradas.items()
        }

        with torch.no_grad():
            saida = modelo_bert(**entradas)

            vetores_tokens = saida.last_hidden_state

            mascara = entradas["attention_mask"].unsqueeze(-1)

            # Ignora padding e tokens especiais nas medias.
            mascara = mascara.expand_as(vetores_tokens).float()
            mascara = mascara.clone()

            ids = entradas["input_ids"]

            especiais = torch.zeros_like(
                entradas["attention_mask"]
            )

            for token_id in tokenizador.all_special_ids:
                especiais |= (ids == token_id).long()

            mascara *= (1 - especiais).unsqueeze(-1).float()

            soma = (vetores_tokens * mascara).sum(dim=1)
            quantidade = mascara.sum(dim=1).clamp(min=1e-9)

            embeddings = soma / quantidade

            # Normalizacao para facilitar a comparacao.
            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

            todos_embeddings.append(
                embeddings.cpu().numpy()
            )

        processados = min(
            inicio + len(lote),
            len(lista_textos)
        )

        print(
            f"\rGerando embeddings BERT: "
            f"{processados}/{len(lista_textos)}",
            end="",
            flush=True
        )

    print()

    return np.concatenate(todos_embeddings, axis=0)


print("\nGerando embeddings BERT das noticias...")

textos_noticias = df["_texto_busca"].tolist()

embeddings_bert = gerar_embeddings_bert(textos_noticias)

dados_bert = df[
    ["titulo", "data_publicacao", "url"]
].copy()

for i in range(embeddings_bert.shape[1]):
    dados_bert[f"embedding_{i}"] = embeddings_bert[:, i]

dados_bert.to_csv(
    ARQUIVO_EMBEDDINGS_BERT,
    index=False,
    encoding="utf-8-sig"
)

print("Arquivo salvo:", ARQUIVO_EMBEDDINGS_BERT)


# ============================================================
# BUSCA WORD2VEC
# ============================================================

def consultar_word2vec(consulta):

    tokens_consulta = tokenizar(consulta)

    tokens_conhecidos = [
        token
        for token in tokens_consulta
        if token in modelo_w2v.wv
    ]

    tokens_desconhecidos = [
        token
        for token in tokens_consulta
        if token not in modelo_w2v.wv
    ]

    print("\n========== WORD2VEC ==========")
    print("Tokens conhecidos:", tokens_conhecidos or "Nenhum")
    print("Tokens fora do vocabulario:", tokens_desconhecidos or "Nenhum")

    if not tokens_conhecidos:
        print(
            "Nao foi possivel representar a consulta "
            "em Word2Vec: nenhum termo conhecido."
        )
        return []

    vetor_consulta = np.mean(
        [
            modelo_w2v.wv[token]
            for token in tokens_conhecidos
        ],
        axis=0
    )

    resultados = []

    for i, vetor_noticia in enumerate(embeddings_w2v):

        if np.linalg.norm(vetor_noticia) == 0:
            continue

        similaridade = similaridade_cosseno(
            vetor_consulta,
            vetor_noticia
        )

        resultados.append((i, similaridade))

    resultados.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return resultados[:QUANTIDADE_RESULTADOS]


# ============================================================
# BUSCA BERTIMBAU
# ============================================================

def consultar_bert(consulta):

    print("\n========== BERTIMBAU ==========")

    vetor_consulta = gerar_embeddings_bert([consulta])[0]

    similaridades = embeddings_bert @ vetor_consulta

    indices = np.argsort(similaridades)[::-1]

    resultados = []

    for i in indices[:QUANTIDADE_RESULTADOS]:
        resultados.append(
            (int(i), float(similaridades[i]))
        )

    return resultados


# ============================================================
# EXIBIR RESULTADOS
# ============================================================

def exibir_resultados(resultados, nome_modelo):

    if not resultados:
        print(f"\nNenhum resultado encontrado em {nome_modelo}.")
        return

    linhas = []

    for posicao, (indice, similaridade) in enumerate(
        resultados,
        start=1
    ):

        linha = df.iloc[indice]

        linhas.append({
            "posicao": posicao,
            "titulo": linha["titulo"],
            "data_publicacao": linha["data_publicacao"],
            "url": linha["url"],
            "similaridade": round(similaridade, 6)
        })

    resultado_df = pd.DataFrame(linhas)

    print(resultado_df.to_string(index=False))


# ============================================================
# RELATORIO TXT
# ============================================================

def salvar_relatorio(
    consulta,
    resultados_w2v,
    resultados_bert
):

    with open(
        ARQUIVO_RELATORIO,
        "a",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write("\n" + "=" * 80 + "\n")
        arquivo.write(f"CONSULTA: {consulta}\n")
        arquivo.write("=" * 80 + "\n")

        for nome_modelo, resultados in [
            ("WORD2VEC", resultados_w2v),
            ("BERTIMBAU", resultados_bert)
        ]:

            arquivo.write(f"\n========== {nome_modelo} ==========\n")

            if not resultados:
                arquivo.write("Nenhum resultado encontrado.\n")
                continue

            for posicao, (indice, similaridade) in enumerate(
                resultados,
                start=1
            ):

                linha = df.iloc[indice]

                arquivo.write(f"\nPosicao: {posicao}\n")
                arquivo.write(
                    f"Titulo: {linha['titulo']}\n"
                )
                arquivo.write(
                    f"Data: {linha['data_publicacao']}\n"
                )
                arquivo.write(
                    f"URL: {linha['url']}\n"
                )
                arquivo.write(
                    f"Similaridade: {similaridade:.6f}\n"
                )

    print("\nRelatorio atualizado:", ARQUIVO_RELATORIO)


# ============================================================
# CONSULTAS CONTINUAS
# ============================================================

print("\n" + "=" * 60)
print("BUSCA SEMANTICA DE NOTICIAS DO GE")
print("Digite uma palavra ou frase para pesquisar.")
print("Digite 'sair' para encerrar.")
print("=" * 60)

while True:

    try:
        consulta = input("\nConsulta (ou 'sair'): ").strip()

        if consulta.casefold() == "sair":
            print("Programa encerrado.")
            break

        if not consulta:
            continue

        resultados_w2v = consultar_word2vec(consulta)

        print("\nMelhores resultados Word2Vec:")
        exibir_resultados(
            resultados_w2v,
            "Word2Vec"
        )

        resultados_bert = consultar_bert(consulta)

        print("\nMelhores resultados BERTimbau:")
        exibir_resultados(
            resultados_bert,
            "BERTimbau"
        )

        salvar_relatorio(
            consulta,
            resultados_w2v,
            resultados_bert
        )

    except KeyboardInterrupt:
        print("\nPrograma encerrado.")
        break

    except Exception as erro:
        print(f"\nErro durante a consulta: {erro}")
