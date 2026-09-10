import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

BASE_FEED = "https://ge.globo.com/futebol/index/feed/pagina-{}.ghtml"


def obter_links_noticias(qtd=100):
    links = set()
    pagina = 1

    while len(links) < qtd:
        print(f"Buscando página {pagina}...")

        try:
            url = BASE_FEED.format(pagina)

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=15
            )

            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            for a in soup.find_all("a", href=True):
                href = a["href"]

                if  (
                    "/noticia/" in href
                    and "/futebol/" in href
                ):
                    links.add(href)

                    if len(links) >= qtd:
                        break

        except Exception as e:
            print(f"Erro na página {pagina}: {e}")

        pagina += 1
        time.sleep(1)

    return list(links)[:qtd]


def limpar_texto(texto):
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def texto_valido(texto):
    texto_lower = texto.lower()

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

    return not any(termo in texto_lower for termo in termos_excluir)


def extrair_noticia(url):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        data_publicacao = ""

        meta_data = soup.find("meta", {"property": "article:published_time"})

        if meta_data:
           data_publicacao = meta_data.get("content", "")

        titulo = ""

        h1 = soup.find("h1")
        if h1:
            titulo = limpar_texto(h1.get_text())

        conteudo = []

        # Principais áreas onde o GE costuma colocar o texto
        seletores = [
            ".mc-article-body",
            ".mc-content",
            ".content-text",
            ".entry-content",
            "article"
        ]

        corpo = None

        for seletor in seletores:
            corpo = soup.select_one(seletor)

            if corpo:
                break

        if corpo:
            paragrafos = corpo.find_all("p")
        else:
            paragrafos = soup.find_all("p")

        for p in paragrafos:
            texto = limpar_texto(p.get_text(" ", strip=True))

            if len(texto) < 40:
                continue

            if texto.startswith("+"):
                continue

            if not texto_valido(texto):
                continue

            conteudo.append(texto)

        texto_final = " ".join(conteudo)

        return {
            "titulo": titulo,
            "data_publicacao": data_publicacao,
            "url": url,
            "texto": texto_final
        }

    except Exception as e:
        print(f"Erro ao ler notícia: {url}")
        print(e)
        return None


def obter_area_trabalho():
    home = os.path.expanduser("~")

    caminhos = [
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "OneDrive", "Área de Trabalho"),
        os.path.join(home, "Área de Trabalho")
    ]

    for caminho in caminhos:
        if os.path.exists(caminho):
            return caminho

    return home


def main():
    links = obter_links_noticias(100)

    print(f"\n{len(links)} links encontrados.\n")

    noticias = []

    for i, link in enumerate(links, start=1):
        print(f"[{i}/{len(links)}] Coletando...")

        noticia = extrair_noticia(link)

        if noticia and noticia["texto"]:
            noticias.append(noticia)

        time.sleep(0.5)

    df = pd.DataFrame(noticias)

    desktop = obter_area_trabalho()

    caminho_csv = os.path.join(
        desktop,
        "noticias_ge.csv"
    )

    df.to_csv(
        caminho_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n--------------------------------")
    print(f"CSV gerado com {len(df)} notícias")
    print(f"Arquivo salvo em:")
    print(caminho_csv)
    print("--------------------------------")


if __name__ == "__main__":
    main()