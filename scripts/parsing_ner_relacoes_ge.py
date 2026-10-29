# python -m pip install spacy pandas networkx matplotlib
# python -m spacy download pt_core_news_sm

from collections import Counter
from pathlib import Path
import ast
import re
import unicodedata
import textwrap
import warnings

import pandas as pd
import spacy
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacy import displacy


# Os scripts ficam em scripts/; a raiz do projeto é a pasta acima.
PASTA_RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = PASTA_RAIZ / "dados" / "processados"
PASTA = PASTA_RAIZ / "resultados" / "parsing_ner"
PASTA.mkdir(parents=True, exist_ok=True)

ARQUIVO_ENTRADA = PASTA_DADOS / "noticias_ge_processado.csv"

# Salvo compactado: com 500 noticias o CSV passa de 100 MB,
# limite de arquivo do GitHub. O pandas le .csv.gz direto.
ARQUIVO_PARSING = PASTA / "noticias_ge_parsing.csv.gz"
ARQUIVO_ENTIDADES = PASTA / "noticias_ge_entidades.csv"
ARQUIVO_RELACOES = PASTA / "noticias_ge_relacoes.csv"

ARQUIVO_GRAFO = PASTA / "noticias_ge_relacoes.gexf"
ARQUIVO_GRAFO_PNG = PASTA / "noticias_ge_grafo.png"

ARQUIVO_RELATORIO = (
    PASTA / "noticias_ge_relatorio_parsing_ner.txt"
)

ARQUIVO_NER_HTML = (
    PASTA / "noticias_ge_ner_visualizacao.html"
)

ARQUIVO_DEP_HTML = (
    PASTA / "noticias_ge_dependencias_visualizacao.html"
)

MODELO_SPACY = "pt_core_news_sm"

# Quantidade maxima de noticias usadas nas visualizacoes HTML.
# Os CSVs continuam incluindo todas as noticias.
MAX_NOTICIAS_VISUALIZACAO = 10

# Limita o tamanho da imagem do grafo.
MAX_PARES_GRAFO = 35

RANDOM_STATE = 42

PALAVRA_CREDITO = r"[A-ZÀ-Ú][\wÀ-ÿ.&'-]+"

PADRAO_LEGENDA_FOTO = re.compile(
    r"\s*[—–-]\s*Foto:\s*"
    r"(?:"
    r"(?:Reprodução|Divulgação|REUTERS|Reuters|AFP|EFE|AP|Getty Images|X)\b"
    r"|"
    rf"{PALAVRA_CREDITO}"
    rf"(?:\s+(?:(?:de|da|do|dos|das)\s+)?{PALAVRA_CREDITO})?"
    r")?"
    r"(?:\s*/\s*[\wÀ-ÿ.&-]+)?"
)


def texto_seguro(valor):
    """Converte valores ausentes em texto vazio."""

    if valor is None:
        return ""

    try:
        if pd.isna(valor):
            return ""
    except (TypeError, ValueError):
        pass

    return str(valor).strip()


def normalizar_nome_coluna(nome):
    """Normaliza nomes para facilitar a identificacao das colunas."""

    nome = unicodedata.normalize("NFKD", str(nome))
    nome = nome.encode("ascii", "ignore").decode("ascii")
    nome = nome.lower()

    return "".join(
        caractere if caractere.isalnum() else "_"
        for caractere in nome
    ).strip("_")


def encontrar_coluna(df, nomes):
    """Encontra uma coluna usando uma lista de nomes alternativos."""

    colunas = {
        normalizar_nome_coluna(coluna): coluna
        for coluna in df.columns
    }

    for nome in nomes:
        chave = normalizar_nome_coluna(nome)

        if chave in colunas:
            return colunas[chave]

    return None


def converter_texto(valor):
    """
    Converte textos e listas de tokens para string.

    Isso permite utilizar tanto o texto original quanto
    uma coluna de tokens, caso o texto nao esteja disponivel.
    """

    if valor is None:
        return ""

    try:
        if pd.isna(valor):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(valor, (list, tuple)):
        return " ".join(str(item) for item in valor)

    texto = str(valor).strip()

    if texto.startswith("[") and texto.endswith("]"):
        try:
            lista = ast.literal_eval(texto)

            if isinstance(lista, (list, tuple)):
                return " ".join(str(item) for item in lista)

        except (ValueError, SyntaxError):
            pass

    return texto



def ler_csv(caminho):
    """Le o CSV tentando codificacoes comuns."""

    ultimo_erro = None

    for codificacao in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                caminho,
                encoding=codificacao,
                sep=None,
                engine="python"
            )

        except Exception as erro:
            ultimo_erro = erro

    raise RuntimeError(
        f"Nao foi possivel ler o arquivo {caminho}.\n"
        f"Erro: {ultimo_erro}"
    )


def carregar_noticias():
    """Carrega e organiza os dados do projeto GE."""

    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {ARQUIVO_ENTRADA}\n"
            "Gere-o com processamento.py "
            "(dados/processados)."
        )

    original = ler_csv(ARQUIVO_ENTRADA)

    coluna_titulo = encontrar_coluna(
        original,
        ["titulo", "title", "headline", "manchete"]
    )

    coluna_texto = encontrar_coluna(
        original,
        [
            "texto",
            "conteudo",
            "corpo",
            "noticia",
            "article_text",
            "content",
            "texto_original",
            "texto_limpo",
            "texto_processado"
        ]
    )

    coluna_tokens = encontrar_coluna(
        original,
        [
            "tokens_sem_stopwords",
            "tokens",
            "tokens_limpos"
        ]
    )

    coluna_url = encontrar_coluna(
        original,
        ["url", "link", "id"]
    )

    coluna_data = encontrar_coluna(
        original,
        [
            "data_publicacao",
            "data",
            "data_publicacao_noticia",
            "published_at"
        ]
    )

    if coluna_titulo is None and coluna_texto is None:
        if coluna_tokens is None:
            raise ValueError(
                "Nao encontrei titulo, texto ou tokens no CSV.\n"
                f"Colunas encontradas: {list(original.columns)}"
            )

    noticias = []

    for indice, (_, linha) in enumerate(original.iterrows(), start=1):

        titulo = (
            converter_texto(linha.get(coluna_titulo, ""))
            if coluna_titulo else ""
        )

        texto = (
            converter_texto(linha.get(coluna_texto, ""))
            if coluna_texto else ""
        )

        tokens = (
            converter_texto(linha.get(coluna_tokens, ""))
            if coluna_tokens else ""
        )

        url = (
            converter_texto(linha.get(coluna_url, ""))
            if coluna_url else ""
        )

        data = (
            converter_texto(linha.get(coluna_data, ""))
            if coluna_data else ""
        )

        if not texto:
            texto = tokens

        if not titulo and not texto:
            continue

        texto = PADRAO_LEGENDA_FOTO.sub(". ", texto)

        if titulo and texto:
            if titulo[-1] not in ".!?":
                titulo_sentenca = titulo + "."
            else:
                titulo_sentenca = titulo

            texto_analise = f"{titulo_sentenca}\n\n{texto}"
        else:
            texto_analise = titulo or texto

        noticia_id = url if url else f"noticia_{indice}"

        noticias.append({
            "noticia_id": noticia_id,
            "titulo": titulo,
            "data_publicacao": data,
            "url": url,
            "texto": texto,
            "texto_analise": texto_analise
        })

    df = pd.DataFrame(noticias)

    if df.empty:
        raise ValueError("O CSV nao contem noticias com texto valido.")

    df["_chave_deduplicacao"] = df.apply(
        lambda linha: (
            "url:" + linha["url"]
            if linha["url"]
            else "texto:" + linha["titulo"] + "|" + linha["texto"]
        ),
        axis=1
    )

    df = df.drop_duplicates(
        subset=["_chave_deduplicacao"],
        keep="first"
    )

    df = df.drop(columns=["_chave_deduplicacao"])

    return df.reset_index(drop=True)


def carregar_modelo():
    """Carrega o modelo de portugues do spaCy."""

    try:
        nlp = spacy.load(MODELO_SPACY)

    except OSError as erro:
        raise RuntimeError(
            "O modelo pt_core_news_sm nao esta instalado.\n"
            "Execute:\n"
            "python -m spacy download pt_core_news_sm"
        ) from erro

    print(
        f"spaCy: {spacy.__version__} | "
        f"Modelo: {nlp.meta.get('version', 'desconhecido')}"
    )

    return nlp


def analisar_dependencias(doc, noticia):
    """
    Registra os tokens e suas relacoes sintaticas.

    ROOT indica a raiz da estrutura.
    nsubj indica sujeito nominal.
    obj indica objeto.
    det indica determinante.
    amod indica modificador adjetival.
    """

    registros = []

    for numero_sentenca, sentenca in enumerate(doc.sents, start=1):

        for token in sentenca:

            registros.append({
                "noticia_id": noticia["noticia_id"],
                "titulo": noticia["titulo"],
                "data_publicacao": noticia["data_publicacao"],
                "url": noticia["url"],
                "numero_sentenca": numero_sentenca,
                "sentenca": sentenca.text.strip(),
                "token": token.text,
                "lema": token.lemma_,
                "classe": token.pos_,
                "tag": token.tag_,
                "dependencia": token.dep_,
                "nucleo": token.head.text,
                "nucleo_lema": token.head.lemma_,
                "indice_token": token.i,
                "inicio_caractere": token.idx,
                "fim_caractere": token.idx + len(token.text)
            })

    return registros


def extrair_entidades(doc, noticia):
    """
    Extrai entidades nomeadas previstas pelo spaCy.

    Exemplos de categorias dependem do modelo:
    PER, LOC, ORG, MISC, entre outras.
    """

    registros = []

    for entidade in doc.ents:

        registros.append({
            "noticia_id": noticia["noticia_id"],
            "titulo": noticia["titulo"],
            "data_publicacao": noticia["data_publicacao"],
            "url": noticia["url"],
            "entidade": entidade.text,
            "categoria": entidade.label_,
            "descricao_categoria": (
                spacy.explain(entidade.label_) or ""
            ),
            "inicio_caractere": entidade.start_char,
            "fim_caractere": entidade.end_char,
            "sentenca": entidade.sent.text.strip()
        })

    return registros


DEPENDENCIAS_SUJEITO = {
    "nsubj",
    "nsubj:pass",
    "nsubjpass",
    "csubj",
    "csubj:pass",
    "csubjpass"
}

DEPENDENCIAS_OBJETO = {
    "obj",
    "dobj",
    "iobj",
    "obl"
}


def obter_sintagma(token):
    """
    Recupera o trecho associado ao token pela arvore de dependencias.
    """

    tokens = sorted(
        [t for t in token.subtree if not t.is_punct],
        key=lambda t: t.i
    )

    if not tokens:
        return token.text

    inicio = min(t.i for t in tokens)
    fim = max(t.i for t in tokens) + 1

    return token.doc[inicio:fim].text.strip()


def obter_argumento(token):
    """
    Primeiro tenta usar uma entidade nomeada que contenha o nucleo.
    Se nao houver entidade, recupera o sintagma da subarvore.
    """

    for entidade in token.doc.ents:

        if entidade.start <= token.i < entidade.end:
            return entidade.text.strip()

    tokens = sorted(
        [t for t in token.subtree if not t.is_punct],
        key=lambda t: t.i
    )

    while tokens and tokens[0].dep_ in {"det", "case", "prep"}:
        tokens.pop(0)

    if not tokens:
        return token.text.strip()

    inicio = min(t.i for t in tokens)
    fim = max(t.i for t in tokens) + 1

    return token.doc[inicio:fim].text.strip()


def normalizar_preposicao(token):
    """Normaliza algumas contracoes para sua preposicao-base."""

    mapa = {
        "no": "em",
        "na": "em",
        "nos": "em",
        "nas": "em",
        "do": "de",
        "da": "de",
        "dos": "de",
        "das": "de",
        "ao": "a",
        "aos": "a",
        "à": "a",
        "às": "a",
        "pelo": "por",
        "pela": "por",
        "pelos": "por",
        "pelas": "por"
    }

    return mapa.get(
        token.text.casefold(),
        token.lemma_.casefold()
    )


def extrair_relacoes(doc, noticia):
    """
    Extrai triplas:
        sujeito -> relacao -> objeto

    Exemplo:
        "O Vasco contratou o jogador."
        Vasco -> contratar -> jogador

    Tambem tenta capturar complementos preposicionados:
        trabalhar em -> Universidade
        negociar com -> clube

    A regra nao cobre todas as estruturas da lingua.
    """

    registros = []

    sentenca_por_token = {}

    for numero, sentenca in enumerate(doc.sents, start=1):

        for token in sentenca:
            sentenca_por_token[token.i] = (
                numero,
                sentenca.text.strip()
            )

    for verbo in doc:

        if verbo.pos_ != "VERB":
            continue

        sujeitos = [
            filho for filho in verbo.children
            if filho.dep_ in DEPENDENCIAS_SUJEITO
        ]

        complementos = [
            filho for filho in verbo.children
            if filho.dep_ in DEPENDENCIAS_OBJETO
        ]

        if not sujeitos or not complementos:
            continue

        numero_sentenca, texto_sentenca = (
            sentenca_por_token.get(verbo.i, ("", ""))
        )

        for complemento in complementos:

            preposicao = next(
                (
                    filho for filho in complemento.children
                    if filho.dep_ in {"case", "prep"}
                ),
                None
            )

            relacao = verbo.lemma_.casefold()

            if (
                complemento.dep_ == "obl"
                and preposicao is not None
            ):
                relacao += "_" + normalizar_preposicao(preposicao)

            for sujeito in sujeitos:

                nome_sujeito = obter_argumento(sujeito)
                nome_objeto = obter_argumento(complemento)

                if not nome_sujeito or not nome_objeto:
                    continue

                if nome_sujeito.casefold() == nome_objeto.casefold():
                    continue

                registros.append({
                    "noticia_id": noticia["noticia_id"],
                    "titulo": noticia["titulo"],
                    "data_publicacao": noticia["data_publicacao"],
                    "url": noticia["url"],
                    "numero_sentenca": numero_sentenca,
                    "sujeito": nome_sujeito,
                    "relacao": relacao,
                    "objeto": nome_objeto,
                    "sentenca": texto_sentenca
                })

    return registros


def gerar_grafo(registros):
    """
    Cada entidade/argumento vira um no.
    Cada relacao extraida vira uma aresta direcionada.

    O GEXF preserva as relacoes para abrir em ferramentas
    como Gephi. A imagem PNG e uma visualizacao resumida.
    """

    grafo = nx.MultiDiGraph()

    for registro in registros:

        sujeito = texto_seguro(registro["sujeito"])
        objeto = texto_seguro(registro["objeto"])
        relacao = texto_seguro(registro["relacao"])

        if not sujeito or not objeto or not relacao:
            continue

        grafo.add_edge(
            sujeito,
            objeto,
            relation=relacao,
            titulo=texto_seguro(registro["titulo"]),
            url=texto_seguro(registro["url"]),
            sentence=texto_seguro(registro["sentenca"])
        )

    if grafo.number_of_edges() == 0:
        print(
            "\nNenhuma relacao foi extraida. "
            "O grafo nao sera gerado."
        )
        return grafo

    nx.write_gexf(grafo, ARQUIVO_GRAFO)

    contagem = Counter()
    rotulos_relacoes = {}

    for origem, destino, dados in grafo.edges(data=True):

        par = (origem, destino)

        contagem[par] += 1

        rotulos_relacoes.setdefault(par, set()).add(
            str(dados.get("relation", ""))
        )

    pares_frequentes = contagem.most_common(MAX_PARES_GRAFO)

    resumo = nx.DiGraph()

    for (origem, destino), frequencia in pares_frequentes:

        resumo.add_edge(
            origem,
            destino,
            frequencia=frequencia,
            relacao=", ".join(
                sorted(rotulos_relacoes[(origem, destino)])
            )
        )

    largura = max(12, min(22, resumo.number_of_nodes() * 0.45))
    altura = max(8, min(16, resumo.number_of_nodes() * 0.30))

    fig, ax = plt.subplots(figsize=(largura, altura))

    posicoes = nx.spring_layout(
        resumo,
        seed=RANDOM_STATE,
        k=1.5
    )

    nx.draw_networkx_nodes(
        resumo,
        posicoes,
        node_size=2200,
        ax=ax
    )

    nomes = {
        no: textwrap.fill(str(no), width=22)
        for no in resumo.nodes
    }

    nx.draw_networkx_labels(
        resumo,
        posicoes,
        labels=nomes,
        font_size=8,
        ax=ax
    )

    nx.draw_networkx_edges(
        resumo,
        posicoes,
        arrows=True,
        arrowsize=16,
        node_size=2200,
        ax=ax
    )

    rotulos = {
        (origem, destino): textwrap.fill(
            dados["relacao"],
            width=18
        )
        for origem, destino, dados in resumo.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        resumo,
        posicoes,
        edge_labels=rotulos,
        font_size=7,
        rotate=False,
        ax=ax
    )

    ax.set_title(
        "Relacoes extraidas das noticias do GE"
    )

    ax.axis("off")
    fig.tight_layout()

    fig.savefig(
        ARQUIVO_GRAFO_PNG,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return grafo


def gerar_visualizacoes(nlp, noticias):
    """
    Gera HTML com entidades e dependencias de uma amostra pequena.
    Os CSVs principais continuam contendo todas as noticias.
    """

    documentos_ner = []
    trechos_dependencias = []

    for noticia in noticias[:MAX_NOTICIAS_VISUALIZACAO]:

        texto = noticia["texto_analise"]

        doc = nlp(texto[:2500])

        documentos_ner.append(doc)

        primeira_sentenca = next(iter(doc.sents), None)

        if primeira_sentenca is not None:
            trechos_dependencias.append(primeira_sentenca)

    if documentos_ner:

        html_ner = displacy.render(
            documentos_ner,
            style="ent",
            page=True
        )

        ARQUIVO_NER_HTML.write_text(
            html_ner,
            encoding="utf-8"
        )

    if trechos_dependencias:

        html_dep = displacy.render(
            trechos_dependencias,
            style="dep",
            page=True,
            options={
                "compact": True,
                "distance": 90
            }
        )

        ARQUIVO_DEP_HTML.write_text(
            html_dep,
            encoding="utf-8"
        )


def gerar_relatorio(
    noticias,
    tokens,
    entidades,
    relacoes,
    grafo
):
    """Gera resumo dos resultados em um arquivo TXT."""

    linhas = [
        "=" * 80,
        "RELATORIO - PARSING, NER E RELACOES DAS NOTICIAS DO GE",
        "=" * 80,
        "",
        f"Arquivo de entrada: {ARQUIVO_ENTRADA.name}",
        f"Noticias processadas: {len(noticias)}",
        f"Tokens analisados: {len(tokens)}",
        f"Entidades reconhecidas: {len(entidades)}",
        f"Relacoes extraidas: {len(relacoes)}",
        f"Nos no grafo: {grafo.number_of_nodes()}",
        f"Arestas no grafo: {grafo.number_of_edges()}",
        "",
        "CATEGORIAS NER RECONHECIDAS",
        "-" * 80
    ]

    if entidades:

        contagem_categorias = Counter(
            registro["categoria"]
            for registro in entidades
        )

        for categoria, quantidade in contagem_categorias.most_common():

            descricao = spacy.explain(categoria) or ""

            linhas.append(
                f"{categoria} ({descricao}): {quantidade}"
            )

    else:
        linhas.append("Nenhuma entidade reconhecida.")

    linhas.extend([
        "",
        "ENTIDADES MAIS FREQUENTES",
        "-" * 80
    ])

    if entidades:

        contagem_entidades = Counter(
            registro["entidade"]
            for registro in entidades
        )

        for nome, quantidade in contagem_entidades.most_common(30):
            linhas.append(f"{nome}: {quantidade}")

    else:
        linhas.append("Sem entidades para listar.")

    linhas.extend([
        "",
        "RELACOES MAIS FREQUENTES",
        "-" * 80
    ])

    if relacoes:

        contagem_relacoes = Counter(
            (
                registro["sujeito"],
                registro["relacao"],
                registro["objeto"]
            )
            for registro in relacoes
        )

        for (sujeito, relacao, objeto), quantidade in (
            contagem_relacoes.most_common(50)
        ):
            linhas.append(
                f"{sujeito} --{relacao}--> {objeto} "
                f"(ocorrencias: {quantidade})"
            )

    else:
        linhas.append(
            "Nenhuma tripla foi extraida pelas regras utilizadas."
        )

    linhas.extend([
        "",
        "LIMITACOES",
        "-" * 80,
        "O parsing e o NER sao previsoes do modelo, nao verificacoes factuais.",
        "As relacoes sao extraidas por regras sobre dependencias sintaticas.",
        "Nem todas as relacoes presentes no texto serao identificadas.",
        "O programa nao resolve correferencia entre nomes e pronomes.",
        "Os resultados precisam ser revisados antes de serem interpretados.",
        "",
        "ARQUIVOS GERADOS",
        "-" * 80,
        ARQUIVO_PARSING.name,
        ARQUIVO_ENTIDADES.name,
        ARQUIVO_RELACOES.name,
        ARQUIVO_RELATORIO.name,
        ARQUIVO_NER_HTML.name if ARQUIVO_NER_HTML.exists() else
        "(visualizacao NER nao gerada)",
        ARQUIVO_DEP_HTML.name if ARQUIVO_DEP_HTML.exists() else
        "(visualizacao de dependencias nao gerada)",
        ARQUIVO_GRAFO.name if ARQUIVO_GRAFO.exists() else
        "(grafo GEXF nao gerado)",
        ARQUIVO_GRAFO_PNG.name if ARQUIVO_GRAFO_PNG.exists() else
        "(imagem do grafo nao gerada)"
    ])

    ARQUIVO_RELATORIO.write_text(
        "\n".join(linhas),
        encoding="utf-8"
    )


def main():

    print("=" * 80)
    print("PARSING, NER E EXTRACAO DE RELACOES - NOTICIAS DO GE")
    print("=" * 80)

    try:
        df = carregar_noticias()
        nlp = carregar_modelo()

    except (FileNotFoundError, ValueError, RuntimeError) as erro:
        print(f"\nErro: {erro}")
        return

    print(f"\nNoticias encontradas: {len(df)}")
    print("Processando os textos...")

    documentos = []

    try:
        for doc in nlp.pipe(
            df["texto_analise"].tolist(),
            batch_size=16
        ):
            documentos.append(doc)

    except Exception as erro:
        print(f"\nErro durante o processamento: {erro}")
        return

    registros_tokens = []
    registros_entidades = []
    registros_relacoes = []

    for indice, (_, noticia) in enumerate(df.iterrows()):

        doc = documentos[indice]

        print(
            f"\rProcessando noticia {indice + 1}/{len(df)}",
            end="",
            flush=True
        )

        metadados = noticia.to_dict()

        registros_tokens.extend(
            analisar_dependencias(doc, metadados)
        )

        registros_entidades.extend(
            extrair_entidades(doc, metadados)
        )

        registros_relacoes.extend(
            extrair_relacoes(doc, metadados)
        )

    print("\n\nSalvando os CSVs...")

    colunas_tokens = [
        "noticia_id",
        "titulo",
        "data_publicacao",
        "url",
        "numero_sentenca",
        "sentenca",
        "token",
        "lema",
        "classe",
        "tag",
        "dependencia",
        "nucleo",
        "nucleo_lema",
        "indice_token",
        "inicio_caractere",
        "fim_caractere"
    ]

    colunas_entidades = [
        "noticia_id",
        "titulo",
        "data_publicacao",
        "url",
        "entidade",
        "categoria",
        "descricao_categoria",
        "inicio_caractere",
        "fim_caractere",
        "sentenca"
    ]

    colunas_relacoes = [
        "noticia_id",
        "titulo",
        "data_publicacao",
        "url",
        "numero_sentenca",
        "sujeito",
        "relacao",
        "objeto",
        "sentenca"
    ]

    df_tokens = pd.DataFrame(
        registros_tokens,
        columns=colunas_tokens
    )

    df_entidades = pd.DataFrame(
        registros_entidades,
        columns=colunas_entidades
    )

    df_relacoes = pd.DataFrame(
        registros_relacoes,
        columns=colunas_relacoes
    )

    df_tokens.to_csv(
        ARQUIVO_PARSING,
        index=False,
        encoding="utf-8-sig"
    )

    df_entidades.to_csv(
        ARQUIVO_ENTIDADES,
        index=False,
        encoding="utf-8-sig"
    )

    df_relacoes.to_csv(
        ARQUIVO_RELACOES,
        index=False,
        encoding="utf-8-sig"
    )

    print("Gerando o grafo...")

    grafo = gerar_grafo(registros_relacoes)

    print("Gerando visualizacoes HTML...")

    try:
        gerar_visualizacoes(nlp, df.to_dict("records"))

    except Exception as erro:
        warnings.warn(
            "Nao foi possivel gerar uma ou mais visualizacoes HTML: "
            + str(erro)
        )

    gerar_relatorio(
        df,
        registros_tokens,
        registros_entidades,
        registros_relacoes,
        grafo
    )

    print("\n" + "=" * 80)
    print("PROCESSAMENTO CONCLUIDO")
    print("=" * 80)

    print(f"\nParsing:   {ARQUIVO_PARSING}")
    print(f"Entidades: {ARQUIVO_ENTIDADES}")
    print(f"Relacoes:  {ARQUIVO_RELACOES}")
    print(f"Relatorio: {ARQUIVO_RELATORIO}")

    if grafo.number_of_edges() > 0:
        print(f"Grafo GEXF: {ARQUIVO_GRAFO}")
        print(f"Grafo PNG:  {ARQUIVO_GRAFO_PNG}")

    if ARQUIVO_NER_HTML.exists():
        print(f"NER HTML:   {ARQUIVO_NER_HTML}")

    if ARQUIVO_DEP_HTML.exists():
        print(f"Dependencias HTML: {ARQUIVO_DEP_HTML}")

    print(
        "\nObservacao: as relacoes sao aproximacoes baseadas em regras. "
        "Revise as triplas antes de trata-las como fatos."
    )


if __name__ == "__main__":
    main()
