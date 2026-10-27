# python -m pip install pandas scikit-learn matplotlib joblib

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# Os scripts ficam em scripts/; a raiz do projeto é a pasta acima.
PASTA_RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = PASTA_RAIZ / "dados" / "processados"
PASTA = PASTA_RAIZ / "resultados" / "classificacao_tfidf"
PASTA.mkdir(parents=True, exist_ok=True)
ARQUIVO_ENTRADA = PASTA_DADOS / "noticias_ge_processado.csv"
ARQUIVO_ROTULACAO = PASTA / "noticias_ge_rotulacao_tfidf.csv"
ARQUIVO_PREVISOES = PASTA / "noticias_ge_tfidf_categorias.csv"
ARQUIVO_RELATORIO = PASTA / "relatorio_tfidf_categorias.txt"
ARQUIVO_MODELO = PASTA / "modelo_tfidf_categorias.joblib"
ARQUIVO_GRAFICO = PASTA / "matriz_confusao_foco_futebol_tfidf.png"

QUANTIDADE_PARA_ROTULAR = 20
MINIMO_PARA_TREINAR = 8
RANDOM_STATE = 42

# Os números servem apenas para agilizar a entrada manual.
TEMAS = {
    "1": (
        "transferencia", "Transferência/mercado da bola",
        "A notícia trata principalmente de mercado da bola, contratação, transferência, negociação, empréstimo, venda, compra ou interesse de um clube em um jogador?"
    ),
    "2": (
        "lesao", "Lesão/problema físico/recuperação",
        "A notícia informa sobre lesão, problema físico, cirurgia, recuperação, departamento médico ou desfalque por motivo físico?"
    ),
    "3": (
        "resultado_partida", "Resultado/placar/desempenho de partida",
        "A notícia relata ou analisa o resultado de uma partida, placar, gols, desempenho durante um jogo ou decisão de uma competição?"
    ),
    "4": (
        "tecnico", "Técnico/treinador/comissão técnica",
        "A notícia trata principalmente de treinador, técnico, comissão técnica, demissão, contratação ou mudança de comando?"
    ),
    "5": (
        "selecao", "Seleção nacional/convocação",
        "A notícia trata de seleção nacional, convocação, jogo de seleção ou participação de jogador/técnico em uma seleção?"
    ),
    "6": (
        "competicao", "Competição/campeonato/classificação/título/rebaixamento",
        "A notícia trata principalmente de uma competição, campeonato, classificação, tabela, rodada, regulamento, título ou rebaixamento?"
    ),
    "7": (
        "declaracao", "Declarações/entrevista/opinião",
        "A notícia destaca declarações, entrevista, coletiva ou opinião de jogador, treinador, dirigente ou outra pessoa do futebol?"
    ),
    "8": (
        "bastidores", "Bastidores/crise/política interna/gestão",
        "A notícia trata principalmente de bastidores, crise, política interna, problemas administrativos ou relacionamento dentro de um clube?"
    ),
    "9": (
        "futebol_feminino", "Futebol feminino",
        "A notícia trata de futebol feminino, equipe feminina, jogadoras ou competições de futebol feminino?"
    ),
    "10": (
        "base", "Categorias de base/futebol juvenil",
        "A notícia trata de categorias de base, futebol juvenil, sub-20, sub-17, formação ou jovens jogadores?"
    ),
}

COLUNAS_TEMAS = [item[0] for item in TEMAS.values()]
COLUNAS_ROTULOS = [f"rotulo_{tema}" for tema in COLUNAS_TEMAS]
COLUNA_SCORE = "rotulo_foco_futebol_score"

DESCRICAO_SCORE = {
    1: "Quase não trata de futebol; a relação é incidental.",
    2: "Futebol é assunto secundário e há pouca informação específica.",
    3: "Trata parcialmente de futebol, mas divide o foco com outros assuntos.",
    4: "Trata diretamente de futebol e apresenta informação relevante.",
    5: "É inteiramente centrada em futebol, clube, jogador, partida ou competição.",
}


def texto_seguro(valor):
    if valor is None or pd.isna(valor):
        return ""
    return str(valor).strip()


def montar_texto(linha):
    titulo = texto_seguro(linha.get("titulo", ""))
    texto = texto_seguro(linha.get("texto", ""))

    if titulo and texto:
        return f"Título: {titulo}\n\nTexto da notícia: {texto}"
    if titulo:
        return titulo
    if texto:
        return texto

    tokens = texto_seguro(linha.get("tokens_sem_stopwords", ""))
    return tokens


def carregar_noticias():
    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Não encontrei {ARQUIVO_ENTRADA.name}.\n"
            "Gere-o com processamento.py (dados/processados)."
        )

    df = pd.read_csv(ARQUIVO_ENTRADA, encoding="utf-8-sig", low_memory=False)

    for coluna in ["titulo", "url"]:
        if coluna not in df.columns:
            raise ValueError(f"A coluna obrigatória '{coluna}' não existe no CSV.")

    if "texto" not in df.columns:
        df["texto"] = ""
    if "data_publicacao" not in df.columns:
        df["data_publicacao"] = ""
    if "id" not in df.columns:
        df["id"] = df["url"]

    df = df.fillna("")
    df["titulo"] = df["titulo"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["texto_modelo"] = df.apply(montar_texto, axis=1)
    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df[df["texto_modelo"].str.strip().str.len() > 0].copy()
    return df.reset_index(drop=True)


def criar_ou_carregar_amostra(df):
    """Cria uma amostra fixa na primeira execução e reaproveita só os rótulos manuais."""

    anotacoes = pd.DataFrame()

    if ARQUIVO_ROTULACAO.exists():
        anotacoes = pd.read_csv(ARQUIVO_ROTULACAO, encoding="utf-8-sig").fillna("")
        # Mantém as anotações anteriores e atualiza título/texto a partir do CSV processado.
        colunas_base = ["id", "titulo", "data_publicacao", "url", "texto_modelo"]
        base = df[colunas_base].drop_duplicates(subset=["url"])
        anotacoes = anotacoes.drop(columns=[c for c in colunas_base if c in anotacoes.columns and c != "url"], errors="ignore")
        anotacoes = base.merge(anotacoes, on="url", how="inner", suffixes=("", "_anotado"))

    # Sem arquivo de rótulos, ou nenhuma notícia rotulada está no CSV atual: cria nova amostra.
    if anotacoes.empty:
        quantidade = min(QUANTIDADE_PARA_ROTULAR, len(df))
        amostra = df.sample(n=quantidade, random_state=RANDOM_STATE).copy()
        anotacoes = amostra[["id", "titulo", "data_publicacao", "url", "texto_modelo"]].copy()
        for coluna in COLUNAS_ROTULOS:
            anotacoes[coluna] = ""
        anotacoes[COLUNA_SCORE] = ""
        salvar_rotulacao(anotacoes)

    for coluna in COLUNAS_ROTULOS:
        if coluna not in anotacoes.columns:
            anotacoes[coluna] = ""
    if COLUNA_SCORE not in anotacoes.columns:
        anotacoes[COLUNA_SCORE] = ""

    anotacoes = anotacoes.fillna("")
    return anotacoes.reset_index(drop=True)


def salvar_rotulacao(df):
    df.to_csv(ARQUIVO_ROTULACAO, index=False, encoding="utf-8-sig")


def rotulo_completo(linha):
    if texto_seguro(linha.get(COLUNA_SCORE, "")) not in {"1", "2", "3", "4", "5"}:
        return False
    return all(texto_seguro(linha.get(coluna, "")).lower() in {"sim", "nao"} for coluna in COLUNAS_ROTULOS)


def mostrar_categorias():
    print("\nCategorias (use os números; uma notícia pode ter várias):")
    for numero, (_, descricao, pergunta) in TEMAS.items():
        print(f"  {numero:>2} - {descricao}")
        print(f"      {pergunta}")
    print("  Nenhuma categoria: digite 'nenhum' ou '-' ")


def interpretar_temas(resposta):
    resposta = resposta.strip().lower()
    if resposta in {"nenhum", "nenhuma", "-", "nen", "0"}:
        return set()

    selecionados = set()
    partes = re.split(r"[,;\s]+", resposta)
    for parte in partes:
        if not parte:
            continue
        if parte not in TEMAS:
            raise ValueError(f"Categoria inválida: {parte}")
        selecionados.add(TEMAS[parte][0])
    return selecionados


def rotular_amostra(amostra):
    pendentes = [
        indice for indice, linha in amostra.iterrows()
        if not rotulo_completo(linha)
    ]

    if not pendentes:
        print("As notícias da amostra já estão rotuladas.")
        return amostra

    print("\n" + "=" * 72)
    print("ROTULAGEM MANUAL — AMOSTRA PEQUENA")
    print("=" * 72)
    print(f"Notícias na amostra: {len(amostra)}")
    print(f"Ainda faltam rotular: {len(pendentes)}")
    mostrar_categorias()

    for numero, indice in enumerate(pendentes, start=1):
        linha = amostra.loc[indice]
        print("\n" + "-" * 72)
        print(f"Notícia pendente {numero}/{len(pendentes)}")
        print("Título:", texto_seguro(linha.get("titulo", "")))
        print("Texto:", texto_seguro(linha.get("texto_modelo", ""))[:500].replace("\n", " "))

        while True:
            resposta = input(
                "Temas presentes (ex.: 3,6,7; ENTER pula; Q salva e termina): "
            ).strip()

            if resposta.lower() == "q":
                salvar_rotulacao(amostra)
                print("Rótulos salvos. Você pode continuar na próxima execução.")
                return amostra

            if not resposta:
                print("Notícia pulada; ela continua pendente para a próxima execução.")
                break

            try:
                temas_selecionados = interpretar_temas(resposta)
                break
            except ValueError as erro:
                print(erro)
                print("Digite apenas os números listados, separados por vírgula.")

        if not resposta or resposta.lower() == "q":
            continue

        print("\nFoco em futebol — escolha de 1 a 5:")
        for nota, descricao in DESCRICAO_SCORE.items():
            print(f"  {nota} - {descricao}")

        while True:
            score = input("Nota de foco em futebol (1-5; Q para parar): ").strip()
            if score.lower() == "q":
                salvar_rotulacao(amostra)
                print("Rótulos salvos. Você pode continuar na próxima execução.")
                return amostra
            if score in {"1", "2", "3", "4", "5"}:
                break
            print("Digite uma nota de 1 a 5.")

        for tema in COLUNAS_TEMAS:
            amostra.at[indice, f"rotulo_{tema}"] = "sim" if tema in temas_selecionados else "nao"

        amostra.at[indice, COLUNA_SCORE] = str(score)
        salvar_rotulacao(amostra)

    print(f"\nArquivo de rotulação salvo: {ARQUIVO_ROTULACAO.name}")
    return amostra


def criar_classificador(y_treino):
    """Evita erro quando uma categoria tem apenas sim ou apenas não no treino."""
    classes = np.unique(y_treino)
    if len(classes) >= 2:
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return DummyClassifier(strategy="constant", constant=classes[0])


def extrair_probabilidade_positiva(modelo, X):
    """Retorna a probabilidade da classe positiva (sim/1), mesmo no caso de classe única."""
    if not hasattr(modelo, "predict_proba"):
        return np.asarray(modelo.predict(X), dtype=float)

    probabilidades = modelo.predict_proba(X)
    classes = list(modelo.classes_)
    if 1 in classes:
        return probabilidades[:, classes.index(1)]
    return np.zeros(X.shape[0], dtype=float)


def executar_treinamento(amostra, noticias):
    rotulados = amostra[amostra.apply(rotulo_completo, axis=1)].copy()
    rotulados[COLUNA_SCORE] = rotulados[COLUNA_SCORE].astype(int)

    if len(rotulados) < MINIMO_PARA_TREINAR:
        pendentes = len(amostra) - len(rotulados)
        print(
            f"\nHá {len(rotulados)} notícias totalmente rotuladas e "
            f"{pendentes} pendentes. São necessárias pelo menos "
            f"{MINIMO_PARA_TREINAR} para treinar."
        )
        print("Continue a rotulação e execute o script novamente.")
        return None

    X = rotulados["texto_modelo"].astype(str)
    indices = np.arange(len(rotulados))
    treino_idx, teste_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    X_treino = X.iloc[treino_idx]
    X_teste = X.iloc[teste_idx]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        max_features=20000
    )

    X_treino_tfidf = vectorizer.fit_transform(X_treino)
    X_teste_tfidf = vectorizer.transform(X_teste)

    modelos_temas = {}
    linhas_relatorio = [
        "RELATÓRIO — CLASSIFICAÇÃO DE NOTÍCIAS DO GE",
        "TF-IDF + Regressão Logística; categorias definidas manualmente",
        "=" * 72,
        f"Notícias rotuladas manualmente: {len(rotulados)}",
        f"Treinamento: {len(treino_idx)}",
        f"Teste: {len(teste_idx)}",
        f"Tamanho máximo configurado para amostra: {QUANTIDADE_PARA_ROTULAR}",
        "MÉTRICAS POR TEMA",
        "-" * 72,
    ]

    y_test_df = rotulados.iloc[teste_idx]
    y_train_df = rotulados.iloc[treino_idx]

    for tema in COLUNAS_TEMAS:
        coluna = f"rotulo_{tema}"
        y_treino = y_train_df[coluna].str.lower().eq("sim").astype(int).to_numpy()
        y_teste = y_test_df[coluna].str.lower().eq("sim").astype(int).to_numpy()

        modelo = criar_classificador(y_treino)
        modelo.fit(X_treino_tfidf, y_treino)
        predicoes = modelo.predict(X_teste_tfidf).astype(int)
        modelos_temas[tema] = modelo

        acc = accuracy_score(y_teste, predicoes)
        precisao = precision_score(y_teste, predicoes, zero_division=0)
        revocacao = recall_score(y_teste, predicoes, zero_division=0)
        f1 = f1_score(y_teste, predicoes, zero_division=0)

        descricao = dict((v[0], v[1]) for v in TEMAS.values())[tema]
        linhas_relatorio.extend([
            f"\n{tema} — {descricao}",
            f"Acurácia: {acc:.3f} | Precision: {precisao:.3f} | Recall: {revocacao:.3f} | F1: {f1:.3f}",
            classification_report(
                y_teste,
                predicoes,
                labels=[0, 1],
                target_names=["não", "sim"],
                digits=3,
                zero_division=0
            ).rstrip(),
        ])

    y_score_treino = y_train_df[COLUNA_SCORE].astype(int).to_numpy()
    y_score_teste = y_test_df[COLUNA_SCORE].astype(int).to_numpy()
    modelo_score = criar_classificador(y_score_treino)
    modelo_score.fit(X_treino_tfidf, y_score_treino)
    pred_score_teste = modelo_score.predict(X_teste_tfidf).astype(int)

    linhas_relatorio.extend([
        "\n" + "=" * 72,
        "FOCO EM FUTEBOL (score de 1 a 5)",
        f"Acurácia: {accuracy_score(y_score_teste, pred_score_teste):.3f}",
        f"Erro absoluto médio: {mean_absolute_error(y_score_teste, pred_score_teste):.3f}",
        classification_report(
            y_score_teste,
            pred_score_teste,
            labels=[1, 2, 3, 4, 5],
            digits=3,
            zero_division=0
        ).rstrip(),
        "\nDISTRIBUIÇÃO DOS RÓTULOS MANUAIS",
    ])

    for tema in COLUNAS_TEMAS:
        coluna = f"rotulo_{tema}"
        contagem_sim = int(rotulados[coluna].str.lower().eq("sim").sum())
        linhas_relatorio.append(f"{tema}: sim={contagem_sim}, não={len(rotulados) - contagem_sim}")

    linhas_relatorio.extend([
        "",
        "LEITURA DO MODELO",
        "Cada tema usa classificação binária sim/não. O foco em futebol usa classes de 1 a 5.",
        "Se uma categoria só aparecer como sim ou só como não no conjunto de treino, é usado um classificador constante para evitar erro de treinamento.",
        "Com 20 notícias, os resultados não são suficientemente robustos para conclusões definitivas. Rotular mais notícias tende a melhorar a avaliação.",
    ])

    # Reajusta os modelos no conjunto completo de notícias rotuladas antes de prever o corpus.
    X_rotulados_tfidf = vectorizer.fit_transform(X)
    modelos_temas_finais = {}
    for tema in COLUNAS_TEMAS:
        y_tema = rotulados[f"rotulo_{tema}"].str.lower().eq("sim").astype(int).to_numpy()
        modelo = criar_classificador(y_tema)
        modelo.fit(X_rotulados_tfidf, y_tema)
        modelos_temas_finais[tema] = modelo

    y_score_completo = rotulados[COLUNA_SCORE].astype(int).to_numpy()
    modelo_score_final = criar_classificador(y_score_completo)
    modelo_score_final.fit(X_rotulados_tfidf, y_score_completo)

    bundle = {
        "vectorizer": vectorizer,
        "modelos_temas": modelos_temas_finais,
        "modelo_foco_futebol": modelo_score_final,
        "temas": COLUNAS_TEMAS,
        "descricao_temas": {v[0]: v[1] for v in TEMAS.values()},
    }
    joblib.dump(bundle, ARQUIVO_MODELO)

    X_todas = noticias["texto_modelo"].astype(str)
    X_todas_tfidf = vectorizer.transform(X_todas)
    saida = noticias[[c for c in ["id", "titulo", "data_publicacao", "url"] if c in noticias.columns]].copy()

    temas_por_noticia = []
    for tema in COLUNAS_TEMAS:
        modelo = modelos_temas_finais[tema]
        pred = modelo.predict(X_todas_tfidf).astype(int)
        prob = extrair_probabilidade_positiva(modelo, X_todas_tfidf)
        saida[f"tfidf_tema_{tema}"] = np.where(pred == 1, "sim", "nao")
        saida[f"tfidf_prob_{tema}"] = np.round(prob, 4)

    score_pred = modelo_score_final.predict(X_todas_tfidf).astype(int)
    saida["tfidf_foco_futebol_score"] = score_pred
    if hasattr(modelo_score_final, "predict_proba"):
        score_prob = modelo_score_final.predict_proba(X_todas_tfidf)
        saida["tfidf_foco_futebol_confianca"] = np.round(score_prob.max(axis=1), 4)
    else:
        saida["tfidf_foco_futebol_confianca"] = 1.0

    def montar_temas_detectados(linha):
        return ", ".join(
            tema for tema in COLUNAS_TEMAS
            if linha.get(f"tfidf_tema_{tema}") == "sim"
        )

    saida["tfidf_temas_detectados"] = saida.apply(montar_temas_detectados, axis=1)
    saida.to_csv(ARQUIVO_PREVISOES, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    labels_score = [1, 2, 3, 4, 5]
    matriz = np.zeros((5, 5), dtype=int)
    for real, previsto in zip(y_score_teste, pred_score_teste):
        matriz[real - 1, previsto - 1] += 1
    disp = __import__("sklearn.metrics", fromlist=["ConfusionMatrixDisplay"]).ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=labels_score
    )
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de confusão — foco em futebol")
    fig.tight_layout()
    fig.savefig(ARQUIVO_GRAFICO, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ARQUIVO_RELATORIO.write_text("\n".join(linhas_relatorio), encoding="utf-8")

    print("\nTreinamento concluído.")
    print(f"Relatório: {ARQUIVO_RELATORIO.name}")
    print(f"Previsões de todas as notícias: {ARQUIVO_PREVISOES.name}")
    print(f"Modelo treinado: {ARQUIVO_MODELO.name}")
    print(f"Matriz de confusão: {ARQUIVO_GRAFICO.name}")
    print(f"Notícias rotuladas usadas: {len(rotulados)}")

    return bundle


def consultar_textos(bundle):
    print("\nConsulta manual. Digite um título/texto; ENTER vazio ou 'sair' encerra.")
    temas_descritos = {v[0]: v[1] for v in TEMAS.values()}

    while True:
        texto = input("\nNotícia: ").strip()
        if not texto or texto.lower() in {"sair", "q", "quit", "exit"}:
            break

        vetor = bundle["vectorizer"].transform([texto])
        detectados = []

        print("\nProbabilidade por tema:")
        for tema in bundle["temas"]:
            modelo = bundle["modelos_temas"][tema]
            pred = int(modelo.predict(vetor)[0])
            prob = float(extrair_probabilidade_positiva(modelo, vetor)[0])
            print(f"  {tema} ({temas_descritos[tema]}): {'sim' if pred else 'não'} — {prob:.1%}")
            if pred == 1:
                detectados.append(tema)

        modelo_score = bundle["modelo_foco_futebol"]
        score = int(modelo_score.predict(vetor)[0])
        if hasattr(modelo_score, "predict_proba"):
            confianca = float(modelo_score.predict_proba(vetor).max())
        else:
            confianca = 1.0

        print("Temas detectados:", ", ".join(detectados) if detectados else "nenhum")
        print(f"Foco em futebol (1 a 5): {score} — {DESCRICAO_SCORE[score]}")
        print(f"Confiança do score: {confianca:.1%}")


def main():
    print("=" * 72)
    print("CLASSIFICAÇÃO DE NOTÍCIAS DO GE — TF-IDF")
    print("=" * 72)

    try:
        noticias = carregar_noticias()
        amostra = criar_ou_carregar_amostra(noticias)
    except Exception as erro:
        print(f"Erro ao carregar os dados: {erro}")
        return

    print(f"\nNotícias no CSV processado: {len(noticias)}")
    print(f"Amostra para rotular manualmente: {len(amostra)}")
    print(f"Arquivo de rótulos manuais: {ARQUIVO_ROTULACAO.name}")

    amostra = rotular_amostra(amostra)
    salvar_rotulacao(amostra)

    rotulados = amostra[amostra.apply(rotulo_completo, axis=1)]
    if len(rotulados) < MINIMO_PARA_TREINAR:
        pendentes = len(amostra) - len(rotulados)
        print(f"\nAinda não há rótulos suficientes: {len(rotulados)} completos; {pendentes} pendentes.")
        print("Continue na próxima execução. As anotações já foram salvas.")
        return

    bundle = executar_treinamento(amostra, noticias)
    if bundle is not None:
        consultar_textos(bundle)


if __name__ == "__main__":
    main()
