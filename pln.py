# pip install requests
# python -m pip install pandas
# python -m pip install beautifulsoup4
# python -m pip install lxml

import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

# Cabeçalho enviado nas requisições para identificar o programa
# como um navegador comum.
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# URL do feed de notícias de futebol do GE.
# O {} será substituído pelo número da página.
BASE_FEED = "https://ge.globo.com/futebol/index/feed/pagina-{}.ghtml"


# ==========================================================
# OBTENÇÃO DOS LINKS DAS NOTÍCIAS
# ==========================================================

def obter_links_noticias(qtd=100):
    """
    Percorre as páginas do feed do GE e coleta os links
    das notícias de futebol.

    Parâmetros:
        qtd: quantidade de notícias que devem ser coletadas.

    Retorna:
        Uma lista contendo os links das notícias encontradas.
    """

    # Utiliza um set para evitar links duplicados.
    links = set()

    pagina = 1

    # Continua buscando páginas até atingir a quantidade desejada.
    while len(links) < qtd:

        print(f"Buscando página {pagina}...")

        try:
            # Monta a URL da página atual.
            url = BASE_FEED.format(pagina)

            # Faz a requisição para o GE.
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=15
            )

            # Gera uma exceção caso a página retorne erro HTTP.
            response.raise_for_status()

            # Converte o HTML recebido em uma estrutura que
            # pode ser pesquisada pelo BeautifulSoup.
            soup = BeautifulSoup(response.text, "lxml")

            # Procura todos os elementos <a> que possuem href.
            for a in soup.find_all("a", href=True):

                href = a["href"]

                # Considera apenas links que sejam notícias
                # relacionadas à seção de futebol.
                if (
                    "/noticia/" in href
                    and "/futebol/" in href
                ):

                    links.add(href)

                    # Para de procurar quando atingir a quantidade
                    # solicitada.
                    if len(links) >= qtd:
                        break

        except Exception as e:
            # Caso ocorra algum erro, mostra a página onde ocorreu.
            print(f"Erro na página {pagina}: {e}")

        # Passa para a próxima página.
        pagina += 1

        # Aguarda um pouco antes da próxima requisição para
        # evitar fazer muitas requisições seguidas.
        time.sleep(1)

    # Retorna somente a quantidade solicitada de links.
    return list(links)[:qtd]


# ==========================================================
# LIMPEZA DO TEXTO
# ==========================================================

def limpar_texto(texto):
    """
    Remove espaços e quebras de linha desnecessárias
    do texto extraído da página.
    """

    # Substitui qualquer sequência de espaços, tabs ou
    # quebras de linha por apenas um espaço.
    texto = re.sub(r"\s+", " ", texto)

    # Remove espaços no início e no final.
    return texto.strip()


# ==========================================================
# VALIDAÇÃO DOS PARÁGRAFOS
# ==========================================================

def texto_valido(texto):
    """
    Verifica se um parágrafo deve ser mantido.

    Alguns textos encontrados nas páginas do GE são
    propagandas, recomendações ou conteúdos que não fazem
    parte da notícia. Esses textos são descartados.
    """

    texto_lower = texto.lower()

    # Termos utilizados para identificar conteúdos que
    # não devem entrar no texto final da notícia.
    termos_excluir = [
        "veja também",
        "leia também",
        "assista",
        "mais do ge",
        "clique aqui",
        "confira",
        "publicidade",
        "conteúdo patrocinado",
        "receba as notícias",
        "siga o ge",
        "g1",
        "globoplay",
        "cartola",
        "compre já seus ingressos",
        "o mercado do cartola",
        "palpite ge",
        "veja os detalhes abaixo",
        "adicione o ge nas suas fontes favoritas",
        "google notícias",
        "google news"
    ]

    # Se qualquer termo aparecer no texto, o parágrafo
    # será considerado inválido.
    return not any(
        termo in texto_lower
        for termo in termos_excluir
    )


# ==========================================================
# EXTRAÇÃO DE UMA NOTÍCIA
# ==========================================================

def extrair_noticia(url):
    """
    Acessa uma notícia e extrai:
        - título
        - data de publicação
        - URL
        - texto da notícia

    Retorna:
        Um dicionário com os dados da notícia ou None
        caso ocorra algum erro.
    """

    try:
        # Acessa a página da notícia.
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()

        # Converte o HTML em uma estrutura pesquisável.
        soup = BeautifulSoup(response.text, "lxml")

        # --------------------------------------------------
        # DATA DE PUBLICAÇÃO
        # --------------------------------------------------

        data_publicacao = ""

        # Procura a informação de data dentro da meta tag
        # article:published_time.
        meta_data = soup.find(
            "meta",
            {"property": "article:published_time"}
        )

        if meta_data:
            data_publicacao = meta_data.get(
                "content",
                ""
            )

        # --------------------------------------------------
        # TÍTULO
        # --------------------------------------------------

        titulo = ""

        # O título normalmente está dentro de uma tag <h1>.
        h1 = soup.find("h1")

        if h1:
            titulo = limpar_texto(
                h1.get_text()
            )

        # --------------------------------------------------
        # CONTEÚDO DA NOTÍCIA
        # --------------------------------------------------

        conteudo = []

        # Principais áreas onde o GE pode colocar o corpo
        # da notícia.
        seletores = [
            ".mc-article-body",
            ".mc-content",
            ".content-text",
            ".entry-content",
            "article"
        ]

        corpo = None

        # Tenta encontrar o conteúdo usando os seletores
        # acima, na ordem em que aparecem.
        for seletor in seletores:

            corpo = soup.select_one(seletor)

            if corpo:
                break

        # Se encontrou o corpo da notícia, procura os
        # parágrafos dentro dele.
        #
        # Caso não encontre, procura todos os <p> da página
        # como alternativa.
        if corpo:
            paragrafos = corpo.find_all("p")
        else:
            paragrafos = soup.find_all("p")

        # Analisa cada parágrafo encontrado.
        for p in paragrafos:

            # Extrai o texto do parágrafo e limpa espaços
            # desnecessários.
            texto = limpar_texto(
                p.get_text(
                    " ",
                    strip=True
                )
            )

            # Ignora textos muito pequenos.
            if len(texto) < 40:
                continue

            # Ignora textos que começam com "+".
            if texto.startswith("+"):
                continue

            # Ignora textos identificados como propaganda,
            # recomendação ou conteúdo externo.
            if not texto_valido(texto):
                continue

            # Adiciona o parágrafo à notícia.
            conteudo.append(texto)

        # Junta todos os parágrafos em um único texto.
        texto_final = " ".join(conteudo)

        # Retorna os dados da notícia.
        return {
            "titulo": titulo,
            "data_publicacao": data_publicacao,
            "url": url,
            "texto": texto_final
        }

    except Exception as e:

        # Mostra qual notícia apresentou erro.
        print(f"Erro ao ler notícia: {url}")
        print(e)

        return None


# ==========================================================
# FUNÇÃO PRINCIPAL
# ==========================================================

def main():
    """
    Executa todo o processo de coleta:

    1. Busca os links das notícias.
    2. Acessa cada notícia.
    3. Extrai os dados.
    4. Cria um DataFrame.
    5. Salva os dados em um arquivo CSV.
    """

    # Quantidade de notícias que serão buscadas.
    links = obter_links_noticias(100)

    print(f"\n{len(links)} links encontrados.\n")

    # Lista que armazenará os dados das notícias.
    noticias = []

    # Processa cada link encontrado.
    for i, link in enumerate(
        links,
        start=1
    ):

        print(
            f"[{i}/{len(links)}] Coletando..."
        )

        # Extrai os dados da notícia.
        noticia = extrair_noticia(link)

        # Só adiciona a notícia se ela foi extraída
        # corretamente e possui texto.
        if noticia and noticia["texto"]:
            noticias.append(noticia)

        # Pequena pausa entre as requisições.
        time.sleep(0.5)

    # Converte a lista de notícias em um DataFrame.
    df = pd.DataFrame(noticias)

    # --------------------------------------------------
    # LOCAL ONDE O CSV SERÁ SALVO
    # --------------------------------------------------

    # Obtém a pasta onde este arquivo .py está localizado.
    pasta_script = os.path.dirname(
        os.path.abspath(__file__)
    )

    # Define o nome do arquivo CSV.
    caminho_csv = os.path.join(
        pasta_script,
        "noticias_ge.csv"
    )

    # Salva o DataFrame no CSV.
    #
    # utf-8-sig é utilizado para facilitar a abertura
    # do arquivo diretamente no Excel.
    df.to_csv(
        caminho_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # RESULTADO
    # --------------------------------------------------

    print("\n--------------------------------")
    print(
        f"CSV gerado com {len(df)} notícias"
    )
    print("Arquivo salvo em:")
    print(caminho_csv)
    print("--------------------------------")


# ==========================================================
# INÍCIO DO PROGRAMA
# ==========================================================

# Garante que a função main() seja executada somente
# quando este arquivo for executado diretamente.
if __name__ == "__main__":
    main()