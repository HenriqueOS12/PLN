# python -m pip install typesafe-sdk pandas
# python jev_ge.py

import os
import time
from getpass import getpass
from pathlib import Path

import pandas as pd
from typesafe_sdk import Noul, Score, TypeSafeClient

PASTA = Path(__file__).resolve().parent
ARQUIVO_CSV = PASTA / "noticias_ge_processado.csv"
ARQUIVO_RESULTADOS = PASTA / "noticias_ge_jev.csv"
ARQUIVO_RELATORIO = PASTA / "relatorio_jev_ge.txt"

LIMITE_NOTICIAS = None

LIMIAR_TEMA = 0.5
INTERVALO_ENTRE_CHAMADAS = 0.2

CONTINUAR_EXECUCAO = True

RANDOM_STATE = 42 

PERGUNTAS_JEV = {
    "transferencia": Noul(
        instructions=(
            "A noticia trata principalmente de mercado da bola, "
            "contratacao, transferencia, negociacao, emprestimo, venda, "
            "compra ou interesse de um clube em um jogador?"
        )
    ),
    "lesao": Noul(
        instructions=(
            "A noticia informa sobre lesao, problema fisico, cirurgia, "
            "recuperacao, departamento medico ou desfalque por motivo fisico?"
        )
    ),
    "resultado_partida": Noul(
        instructions=(
            "A noticia relata ou analisa o resultado de uma partida, "
            "placar, gols, desempenho durante um jogo ou decisao de uma competicao?"
        )
    ),
    "tecnico": Noul(
        instructions=(
            "A noticia trata principalmente de treinador, tecnico, "
            "comissao tecnica, demissao, contratacao ou mudanca de comando?"
        )
    ),
    "selecao": Noul(
        instructions=(
            "A noticia trata de selecao nacional, convocacao, jogo de selecao "
            "ou participacao de jogador/tecnico em uma selecao?"
        )
    ),
    "competicao": Noul(
        instructions=(
            "A noticia trata principalmente de uma competicao, campeonato, "
            "classificacao, tabela, rodada, regulamento, titulo ou rebaixamento?"
        )
    ),
    "declaracao": Noul(
        instructions=(
            "A noticia destaca declaracoes, entrevista, coletiva ou opiniao "
            "de jogador, treinador, dirigente ou outra pessoa do futebol?"
        )
    ),
    "bastidores": Noul(
        instructions=(
            "A noticia trata principalmente de bastidores, crise, politica "
            "interna, problemas administrativos ou relacionamento dentro de um clube?"
        )
    ),
    "futebol_feminino": Noul(
        instructions=(
            "A noticia trata de futebol feminino, equipe feminina, jogadoras "
            "ou competicoes de futebol feminino?"
        )
    ),
    "base": Noul(
        instructions=(
            "A noticia trata de categorias de base, futebol juvenil, sub-20, "
            "sub-17, formacao ou jovens jogadores?"
        )
    ),
    "foco_futebol": Score(
        instructions=(
            "Avalie o quanto o conteudo da noticia trata diretamente de futebol. "
            "Considere o texto e o titulo, e nao apenas a presenca de nomes de clubes."
        ),
        criteria=[
            "O conteudo praticamente nao trata de futebol; a relacao e incidental.",
            "O futebol aparece como assunto secundario, com pouca informacao especifica.",
            "A noticia trata parcialmente de futebol, mas divide o foco com outros assuntos.",
            "A noticia trata diretamente de futebol e apresenta informacoes relevantes.",
            "A noticia e inteiramente centrada em futebol, clube, jogador, partida ou competicao."
        ]
    ),
}

def texto_seguro(valor):
    """Converte valores ausentes em texto vazio."""
    if valor is None or pd.isna(valor):
        return ""
    return str(valor).strip()


def montar_texto_noticia(linha):
    """Combina titulo e texto, mantendo o titulo em destaque."""
    titulo = texto_seguro(linha.get("titulo", ""))
    texto = texto_seguro(linha.get("texto", ""))

    if titulo and texto:
        return "Titulo: " + titulo + "\n\nTexto da noticia: " + texto
    return titulo or texto


def valor_float(valor):
    """Converte um valor numerico para float; retorna None se nao for possivel."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def classificar_com_jev(texto, client):
    """Executa uma chamada Jev com todas as perguntas definidas acima."""
    resposta = client.system_one(
        state=texto,
        questions=PERGUNTAS_JEV
    )

    resultado = {}

    for tema in [
        "transferencia",
        "lesao",
        "resultado_partida",
        "tecnico",
        "selecao",
        "competicao",
        "declaracao",
        "bastidores",
        "futebol_feminino",
        "base",
    ]:
        probabilidade = float(resposta.answers[tema].noul)
        resultado["jev_prob_" + tema] = probabilidade
        resultado["jev_tema_" + tema] = (
            "sim" if probabilidade >= LIMIAR_TEMA else "nao"
        )

    resposta_foco = resposta.answers["foco_futebol"]
    resultado["jev_foco_futebol_score"] = (
        valor_float(resposta_foco.score)
    )
    resultado["jev_foco_futebol_confianca"] = (
        valor_float(resposta_foco.confidence)
    )

    probabilidades = getattr(resposta_foco, "probabilities", None)
    if probabilidades is not None:
        for nivel, probabilidade in dict(probabilidades).items():
            try:
                nivel_numero = int(nivel) + 1
            except (TypeError, ValueError):
                nivel_numero = str(nivel)
            resultado["jev_foco_prob_" + str(nivel_numero)] = (
                valor_float(probabilidade)
            )

    temas_detectados = [
        tema for tema in [
            "transferencia", "lesao", "resultado_partida", "tecnico",
            "selecao", "competicao", "declaracao", "bastidores",
            "futebol_feminino", "base"
        ]
        if resultado.get("jev_tema_" + tema) == "sim"
    ]
    resultado["jev_temas_detectados"] = ", ".join(temas_detectados)
    resultado["erro_jev"] = ""
    return resultado


def carregar_noticias():
    """Le o CSV processado e valida as colunas necessarias."""
    if not ARQUIVO_CSV.exists():
        raise FileNotFoundError(
            "Arquivo nao encontrado: " + str(ARQUIVO_CSV) + "\n"
            "Deixe jev_ge.py na mesma pasta de noticias_ge_processado.csv."
        )

    df = pd.read_csv(ARQUIVO_CSV, encoding="utf-8-sig", low_memory=False)

    for coluna in ["titulo", "url"]:
        if coluna not in df.columns:
            raise ValueError(
                "A coluna obrigatoria '" + coluna + "' nao existe no CSV."
            )

    if "texto" not in df.columns:
        df["texto"] = ""
    if "data_publicacao" not in df.columns:
        df["data_publicacao"] = ""

    df = df.fillna("")
    df["titulo"] = df["titulo"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["_texto_jev"] = df.apply(montar_texto_noticia, axis=1)

    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df[df["_texto_jev"].str.strip().str.len() > 0].copy()

    if LIMITE_NOTICIAS is not None:
        df = df.head(LIMITE_NOTICIAS).copy()

    return df.reset_index(drop=True)


def carregar_resultados_existentes():
    """Carrega resultados anteriores, se houver, para retomar a execucao."""
    if CONTINUAR_EXECUCAO and ARQUIVO_RESULTADOS.exists():
        try:
            return pd.read_csv(ARQUIVO_RESULTADOS, encoding="utf-8-sig").fillna("")
        except Exception as erro:
            print("Aviso: nao foi possivel ler o CSV de resultados existente:", erro)
    return pd.DataFrame()


def salvar_resultados(resultados):
    """Salva o progresso em CSV a cada noticia."""
    pd.DataFrame(resultados).to_csv(
        ARQUIVO_RESULTADOS,
        index=False,
        encoding="utf-8-sig"
    )


def gerar_relatorio(df_resultados):
    """Gera um relatorio TXT com resumo e detalhes de cada noticia."""
    linhas = []
    linhas.append("=" * 80)
    linhas.append("RELATORIO - CLASSIFICACAO DE NOTICIAS DO GE COM JEV")
    linhas.append("=" * 80)
    linhas.append("")
    linhas.append("Arquivo de entrada: " + str(ARQUIVO_CSV.name))
    linhas.append("Total de noticias no resultado: " + str(len(df_resultados)))
    linhas.append("Limiar dos temas: " + str(LIMIAR_TEMA))
    linhas.append("")

    erros = 0
    if "erro_jev" in df_resultados.columns:
        erros = int(
            df_resultados["erro_jev"].astype(str).str.strip().ne("").sum()
        )
    linhas.append("Noticias com erro na API: " + str(erros))
    linhas.append("Noticias processadas sem erro: " + str(len(df_resultados) - erros))
    linhas.append("")
    linhas.append("QUANTIDADE DE NOTICIAS POR TEMA DETECTADO")
    linhas.append("-" * 80)

    temas = [
        "transferencia", "lesao", "resultado_partida", "tecnico",
        "selecao", "competicao", "declaracao", "bastidores",
        "futebol_feminino", "base"
    ]

    for tema in temas:
        coluna = "jev_tema_" + tema
        if coluna in df_resultados.columns:
            quantidade = int(
                (df_resultados[coluna].astype(str).str.lower() == "sim").sum()
            )
            linhas.append(f"{tema}: {quantidade}")

    linhas.append("")
    linhas.append("DETALHES DAS NOTICIAS")
    linhas.append("=" * 80)

    for _, linha in df_resultados.iterrows():
        linhas.append("")
        linhas.append("TITULO: " + texto_seguro(linha.get("titulo", "")))
        linhas.append("DATA: " + texto_seguro(linha.get("data_publicacao", "")))
        linhas.append("URL: " + texto_seguro(linha.get("url", "")))
        linhas.append("TEMAS: " + texto_seguro(linha.get("jev_temas_detectados", "")))
        linhas.append(
            "FOCO EM FUTEBOL (score): "
            + texto_seguro(linha.get("jev_foco_futebol_score", ""))
        )
        linhas.append(
            "CONFIANCA DO SCORE: "
            + texto_seguro(linha.get("jev_foco_futebol_confianca", ""))
        )

        for tema in temas:
            coluna_prob = "jev_prob_" + tema
            coluna_simnao = "jev_tema_" + tema
            if coluna_prob in df_resultados.columns:
                linhas.append(
                    f"  {tema}: {texto_seguro(linha.get(coluna_simnao, ''))} "
                    f"(probabilidade={texto_seguro(linha.get(coluna_prob, ''))})"
                )

        erro = texto_seguro(linha.get("erro_jev", ""))
        if erro:
            linhas.append("ERRO: " + erro)

        linhas.append("-" * 80)

    with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))


def executar_classificacao():
    """Processa o corpus, retomando os resultados que ja estao no CSV."""
    df = carregar_noticias()
    existentes = carregar_resultados_existentes()

    urls_concluidas = set()
    resultados = []

    if not existentes.empty and "url" in existentes.columns:
        resultados = existentes.to_dict("records")
        if "erro_jev" in existentes.columns:
            urls_concluidas = set(
                existentes.loc[
                    existentes["erro_jev"].astype(str).str.strip().eq(""),
                    "url"
                ].astype(str)
            )
        else:
            urls_concluidas = set(existentes["url"].astype(str))

    pendentes = df[~df["url"].astype(str).isin(urls_concluidas)].copy()

    print("Noticias encontradas:", len(df))
    print("Noticias ja concluidas:", len(urls_concluidas))
    print("Noticias a processar:", len(pendentes))

    if pendentes.empty:
        print("Nao ha noticias novas para processar.")
        if resultados:
            gerar_relatorio(pd.DataFrame(resultados))
            print("Relatorio atualizado:", ARQUIVO_RELATORIO.name)
        return

    #
    if not os.environ.get("TYPESAFE_API_KEY"):
        os.environ["TYPESAFE_API_KEY"] = getpass(
            "Digite sua TYPESAFE_API_KEY: "
        ).strip()

    if not os.environ.get("TYPESAFE_API_KEY"):
        raise ValueError("A chave TYPESAFE_API_KEY nao foi informada.")

    with TypeSafeClient() as client:
        for posicao, (_, linha) in enumerate(pendentes.iterrows(), start=1):
            print(
                f"\n[{posicao}/{len(pendentes)}] "
                + texto_seguro(linha.get("titulo", ""))[:100]
            )

            resultado_base = {
                "id": texto_seguro(linha.get("id", linha.get("url", ""))),
                "titulo": texto_seguro(linha.get("titulo", "")),
                "data_publicacao": texto_seguro(linha.get("data_publicacao", "")),
                "url": texto_seguro(linha.get("url", "")),
                "texto": texto_seguro(linha.get("texto", "")),
            }

            try:
                resultado_jev = classificar_com_jev(
                    linha["_texto_jev"],
                    client
                )
                resultado_base.update(resultado_jev)
                print("Temas:", resultado_base.get("jev_temas_detectados", "(nenhum acima do limiar)"))
            except Exception as erro:
                resultado_base["erro_jev"] = str(erro)
                print("Erro nesta noticia:", erro)

            resultados = [
                r for r in resultados
                if str(r.get("url", "")) != resultado_base["url"]
            ]
            resultados.append(resultado_base)
            salvar_resultados(resultados)
            gerar_relatorio(pd.DataFrame(resultados))

            time.sleep(INTERVALO_ENTRE_CHAMADAS)

    print("\nProcessamento concluido.")
    print("CSV salvo em:", ARQUIVO_RESULTADOS)
    print("Relatorio TXT salvo em:", ARQUIVO_RELATORIO)


def consulta_manual():
    """Permite classificar textos digitados sem reiniciar o programa."""
    if not os.environ.get("TYPESAFE_API_KEY"):
        os.environ["TYPESAFE_API_KEY"] = getpass(
            "Digite sua TYPESAFE_API_KEY: "
        ).strip()

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Chave nao informada. Consulta manual cancelada.")
        return

    print("\nConsulta manual do Jev. Digite 'sair' para encerrar.")
    with TypeSafeClient() as client:
        while True:
            texto = input("\nDigite o texto ou noticia: ").strip()
            if texto.casefold() in {"sair", "exit", "quit"}:
                break
            if not texto:
                print("Digite algum texto.")
                continue

            try:
                resultado = classificar_com_jev(texto, client)
                print("\nTemas detectados:", resultado["jev_temas_detectados"])
                print(
                    "Foco em futebol (score):",
                    resultado.get("jev_foco_futebol_score")
                )
                print(
                    "Confianca do score:",
                    resultado.get("jev_foco_futebol_confianca")
                )
                print("\nProbabilidades por tema:")
                for tema in [
                    "transferencia", "lesao", "resultado_partida", "tecnico",
                    "selecao", "competicao", "declaracao", "bastidores",
                    "futebol_feminino", "base"
                ]:
                    print(
                        f"  {tema}: {resultado['jev_prob_' + tema]:.3f} "
                        f"-> {resultado['jev_tema_' + tema]}"
                    )
            except Exception as erro:
                print("Erro ao consultar o Jev:", erro)


def main():
    print("=== PLN: noticias do GE classificadas com Jev ===")
    print("1 - Processar/continuar classificacao do CSV")
    print("2 - Fazer consultas manuais continuamente")
    print("0 - Sair")

    while True:
        opcao = input("\nEscolha uma opcao: ").strip()

        if opcao == "1":
            executar_classificacao()
        elif opcao == "2":
            consulta_manual()
        elif opcao == "0":
            print("Encerrado.")
            break
        else:
            print("Opcao invalida. Escolha 1, 2 ou 0.")


if __name__ == "__main__":
    main()
