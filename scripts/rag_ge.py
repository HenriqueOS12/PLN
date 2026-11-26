# python -m pip install sentence-transformers transformers torch numpy pandas scikit-learn nltk
# Com GPU (ex.: Google Colab): python -m pip install accelerate bitsandbytes
#
# Uso:
#   python scripts/rag_ge.py            # perguntas digitadas
#   python scripts/rag_ge.py --teste    # perguntas de teste fixas
#   python scripts/rag_ge.py --avaliar  # avalia a busca com o gabarito

import argparse
import re
from datetime import datetime
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PASTA_RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_ENTRADA = PASTA_RAIZ / "dados" / "processados" / "noticias_ge_processado.csv"
PASTA_SAIDA = PASTA_RAIZ / "resultados" / "rag"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Se a base de notícias ou MAX_CARACTERES_CHUNK mudar, apague estes dois arquivos.
ARQUIVO_CHUNKS = PASTA_SAIDA / "chunks.csv"
ARQUIVO_EMBEDDINGS = PASTA_SAIDA / "embeddings_chunks.npy"
ARQUIVO_RELATORIO = PASTA_SAIDA / "relatorio_rag.txt"
ARQUIVO_AVALIACAO = PASTA_SAIDA / "avaliacao_busca.csv"

MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_GERACAO_GPU = "Qwen/Qwen2.5-7B-Instruct"
MODELO_GERACAO_CPU = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_CARACTERES_CHUNK = 400
TOP_K = 3
PESO_DENSO = 0.7
MAX_NOVOS_TOKENS = 120

RESPOSTA_ABSTENCAO = "Não sei com base nos documentos."

PERGUNTAS_CALIBRACAO = [
    ("Por que a torcida do Corinthians protestou?", True),
    ("Quais jogadores do Flamengo foram convocados para a Seleção?", True),
    ("Qual foi o resultado de Palmeiras x Botafogo?", True),
    ("Quem é o técnico do Corinthians?", True),
    ("Qual é a capital do Canadá?", False),
    ("Como preparar uma lasanha?", False),
    ("Quem ganhou a Copa do Mundo de 1994?", False),
    ("Qual é a fórmula química da água?", False),
]

PERGUNTAS_TESTE = [
    "Quantos gols o Samuel Lino tem no ano?",
    "Quem marcou o gol da vitória do Flamengo sobre o Remo?",
    "O que aconteceu com o mosaico do Corinthians?",
    "Qual é a capital da Austrália?",
]

GABARITO_BUSCA = [
    ("Quantos gols o Samuel Lino tem no ano?", r"12 neste ano|12 gols marcados"),
    ("Quem marcou o gol da vitória do Flamengo sobre o Remo?", r"Samuel Lino comemora o gol da vitória do Flamengo sobre o Remo"),
    ("O que aconteceu com o mosaico do Corinthians?", r"mosaico"),
    ("Qual foi o resultado de Palmeiras x Botafogo?", r"empate do (?:Botafogo|Palmeiras) com o (?:Palmeiras|Botafogo)|(?:Palmeiras|Botafogo)[^.]{0,60}\b0 a 0"),
    ("Quem é o técnico do Flamengo?", r"Leonardo Jardim"),
    ("Por que a torcida do Corinthians protestou?", r"protest"),
    ("Quais jogadores do Flamengo foram convocados para a Seleção?", r"Pedro e Samuel Lino|Samuel Lino e Pedro"),
    ("Quem é o técnico do Corinthians?", r"Diniz"),
    ("Quantos ingressos já foram ocupados para o Athletiba?", r"30 mil ingressos"),
    ("Onde será a final da Copa do Brasil de 2026?", r"Brasília"),
]


for caminho_nltk, recurso in [("corpora/stopwords", "stopwords")]:
    try:
        nltk.data.find(caminho_nltk)
    except LookupError:
        nltk.download(recurso, quiet=True)

df = pd.read_csv(
    ARQUIVO_ENTRADA,
    encoding="utf-8-sig"
).fillna("")


def criar_chunks(noticias, max_caracteres=MAX_CARACTERES_CHUNK):
    chunks = []

    for indice, noticia in noticias.iterrows():
        texto = " ".join(noticia["texto"].split())
        frases = re.split(r"(?<=[.!?])\s+", texto)
        atual = ""
        partes = []

        for frase in frases:
            if atual and len(atual) + 1 + len(frase) > max_caracteres:
                partes.append(atual)
                atual = frase
            else:
                atual = f"{atual} {frase}".strip()

        if atual:
            partes.append(atual)

        for i, trecho in enumerate(partes, start=1):
            chunks.append({
                "chunk_id": f"N{indice + 1}-C{i}",
                "titulo": noticia["titulo"],
                "data": noticia["data_publicacao"],
                "url": noticia["url"],
                "texto": trecho,
            })

    return pd.DataFrame(chunks)


modelo_embeddings = SentenceTransformer(MODELO_EMBEDDINGS, device="cpu")


def carregar_indice():

    if ARQUIVO_CHUNKS.exists() and ARQUIVO_EMBEDDINGS.exists():
        return (
            pd.read_csv(ARQUIVO_CHUNKS, encoding="utf-8-sig").fillna(""),
            np.load(ARQUIVO_EMBEDDINGS)
        )

    chunks = criar_chunks(df)

    print(f"Gerando embeddings de {len(chunks)} chunks (só na primeira execução)...")

    matriz = modelo_embeddings.encode(
        (chunks["titulo"] + ". " + chunks["texto"]).tolist(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    chunks.to_csv(ARQUIVO_CHUNKS, index=False, encoding="utf-8-sig")
    np.save(ARQUIVO_EMBEDDINGS, matriz)

    return chunks, matriz


chunks, matriz_embeddings = carregar_indice()

tfidf = TfidfVectorizer(stop_words=stopwords.words("portuguese"))
matriz_tfidf = tfidf.fit_transform((chunks["titulo"] + ". " + chunks["texto"]).tolist())

print(f"Chunks: {len(chunks)} | matriz de embeddings: {matriz_embeddings.shape}")


def escala_0_1(valores):
    minimo, maximo = valores.min(), valores.max()

    if maximo == minimo:
        return np.zeros_like(valores)

    return (valores - minimo) / (maximo - minimo)


def notas_busca(pergunta):
    vetor_pergunta = modelo_embeddings.encode(
        [pergunta], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    densa = matriz_embeddings @ vetor_pergunta
    lexica = (matriz_tfidf @ normalize(tfidf.transform([pergunta])).T).toarray().ravel()
    hibrida = PESO_DENSO * escala_0_1(densa) + (1 - PESO_DENSO) * escala_0_1(lexica)
    return densa, hibrida


def recuperar(pergunta, top_k=TOP_K):
    densa, hibrida = notas_busca(pergunta)
    indices = np.argsort(hibrida)[::-1][:top_k]
    candidatos = [
        {**chunks.iloc[i].to_dict(),
         "similaridade": float(densa[i]),
         "nota_hibrida": float(hibrida[i])}
        for i in indices
    ]
    return candidatos, float(densa.max())


def calibrar_limiar():
    amostras = []

    for texto, relevante in PERGUNTAS_CALIBRACAO:
        _, maxima = recuperar(texto, top_k=1)
        amostras.append((texto, relevante, maxima))

    positivos = [score for _, rotulo, score in amostras if rotulo]
    negativos = [score for _, rotulo, score in amostras if not rotulo]

    limiar = (max(negativos) + min(positivos)) / 2
    sobreposicao = max(negativos) >= min(positivos)

    return limiar, amostras, sobreposicao


dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

if dispositivo == "cuda":

    MODELO_GERACAO = MODELO_GERACAO_GPU

    modelo_geracao = AutoModelForCausalLM.from_pretrained(
        MODELO_GERACAO,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ),
        device_map="auto"
    )

else:

    MODELO_GERACAO = MODELO_GERACAO_CPU

    modelo_geracao = AutoModelForCausalLM.from_pretrained(
        MODELO_GERACAO,
        dtype=torch.float32,
    )

tokenizer = AutoTokenizer.from_pretrained(MODELO_GERACAO)

modelo_geracao.eval()

print("Modelo de geração:", MODELO_GERACAO, "| dispositivo:", dispositivo)


def construir_prompt(pergunta, trechos):
    contexto = "\n".join(
        f"[{t['chunk_id']}] {t['titulo']}: {t['texto']}" for t in trechos
    )
    return (
        "Responda sempre em português, de forma breve, utilizando somente "
        "as informações dos trechos fornecidos. "
        "Se os trechos não contiverem a resposta, responda exatamente: "
        f"{RESPOSTA_ABSTENCAO}\n\n"
        f"Trechos recuperados:\n{contexto}\n\n"
        f"Pergunta: {pergunta}"
    )


def gerar_resposta(prompt, max_novos_tokens=MAX_NOVOS_TOKENS):
    mensagens = [
        {"role": "system", "content": "Você é um assistente que responde em português usando somente o contexto fornecido."},
        {"role": "user", "content": prompt},
    ]
    entradas = tokenizer.apply_chat_template(
        mensagens, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(modelo_geracao.device)

    input_ids = entradas["input_ids"]

    with torch.inference_mode():
        ids = modelo_geracao.generate(
            input_ids,
            attention_mask=entradas["attention_mask"],
            max_new_tokens=max_novos_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    novos_tokens = ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(novos_tokens, skip_special_tokens=True).strip()


def perguntar_rag(pergunta, limiar, top_k=TOP_K):
    candidatos, maxima = recuperar(pergunta, top_k=top_k)

    if maxima < limiar:
        return {"pergunta": pergunta, "resposta": RESPOSTA_ABSTENCAO, "maxima": maxima,
                "fontes": [], "candidatos": candidatos, "gerou": False}

    resposta = gerar_resposta(construir_prompt(pergunta, candidatos))

    return {"pergunta": pergunta, "resposta": resposta, "maxima": maxima,
            "fontes": candidatos, "candidatos": candidatos, "gerou": True}


def mostrar_resultado(resultado):
    print("\nPERGUNTA:", resultado["pergunta"])
    print("RESPOSTA:", resultado["resposta"])
    print(f"GERADOR EXECUTADO: {resultado['gerou']} (similaridade máxima {resultado['maxima']:.3f})")

    for t in resultado["candidatos"]:
        print(f"  [{t['chunk_id']}] similaridade {t['similaridade']:.3f} | nota híbrida {t['nota_hibrida']:.3f} | {t['titulo']}")


def avaliar_busca():
    linhas = []

    for pergunta, padrao in GABARITO_BUSCA:
        densa, hibrida = notas_busca(pergunta)
        relevantes = chunks["texto"].str.contains(padrao, case=False, regex=True).to_numpy()
        linha = {"pergunta": pergunta, "trechos_relevantes": int(relevantes.sum())}

        for nome, notas in [("densa", densa), ("hibrida", hibrida)]:
            ordem = np.argsort(notas)[::-1]
            linha[f"posicao_{nome}"] = int(np.argmax(relevantes[ordem])) + 1 if relevantes.any() else None

        linhas.append(linha)

    avaliacao = pd.DataFrame(linhas)
    avaliacao.to_csv(ARQUIVO_AVALIACAO, index=False, encoding="utf-8-sig")

    resumo = [
        f"{nome}: top-{TOP_K} {(avaliacao[f'posicao_{nome}'] <= TOP_K).sum()}/{len(avaliacao)} | "
        f"top-5 {(avaliacao[f'posicao_{nome}'] <= 5).sum()}/{len(avaliacao)} | "
        f"posicao media {avaliacao[f'posicao_{nome}'].mean():.1f}"
        for nome in ["densa", "hibrida"]
    ]

    print(avaliacao.to_string(index=False))
    print()
    print("\n".join(resumo))

    with open(ARQUIVO_RELATORIO, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n" + "#" * 80 + "\n")
        arquivo.write(f"AVALIACAO DA BUSCA ({datetime.now():%d/%m/%Y %H:%M:%S}, peso denso {PESO_DENSO})\n")
        arquivo.write("#" * 80 + "\n")
        arquivo.write(avaliacao.to_string(index=False) + "\n\n")
        arquivo.write("\n".join(resumo) + "\n")

    print("\nAvaliacao salva em:", ARQUIVO_AVALIACAO)


def salvar_calibracao(limiar, amostras, sobreposicao):

    with open(ARQUIVO_RELATORIO, "a", encoding="utf-8") as arquivo:

        arquivo.write("\n" + "#" * 80 + "\n")
        arquivo.write(f"CALIBRACAO DO LIMIAR ({datetime.now():%d/%m/%Y %H:%M:%S})\n")
        arquivo.write("#" * 80 + "\n")

        for texto, relevante, score in amostras:
            arquivo.write(f"{score:.3f} | esperado={relevante!s:5} | {texto}\n")

        arquivo.write(f"Limiar: {limiar:.3f}\n")

        if sobreposicao:
            arquivo.write("ATENCAO: as classes se sobrepoem; nenhum limiar separa todos os exemplos.\n")


def salvar_relatorio(resultado, limiar):

    with open(ARQUIVO_RELATORIO, "a", encoding="utf-8") as arquivo:

        arquivo.write("\n" + "=" * 80 + "\n")
        arquivo.write(f"PERGUNTA: {resultado['pergunta']}\n")
        arquivo.write("=" * 80 + "\n")
        arquivo.write(f"Data: {datetime.now():%d/%m/%Y %H:%M:%S}\n")
        arquivo.write(f"Modelo de geracao: {MODELO_GERACAO} ({dispositivo})\n")
        arquivo.write(f"Modelo de embeddings: {MODELO_EMBEDDINGS}\n")
        arquivo.write(f"Busca: hibrida (peso denso {PESO_DENSO})\n")
        arquivo.write(f"Limiar: {limiar:.3f} | similaridade maxima: {resultado['maxima']:.3f}\n")
        arquivo.write(f"Decisao: {'GERAR' if resultado['gerou'] else 'ABSTER-SE'}\n")

        arquivo.write("\n========== RESPOSTA ==========\n")
        arquivo.write(f"{resultado['resposta']}\n")

        arquivo.write("\n========== TRECHOS CANDIDATOS ==========\n")

        for posicao, trecho in enumerate(resultado["candidatos"], start=1):
            arquivo.write(f"\nPosicao: {posicao}\n")
            arquivo.write(f"Chunk: {trecho['chunk_id']}\n")
            arquivo.write(f"Titulo: {trecho['titulo']}\n")
            arquivo.write(f"Data: {trecho['data']}\n")
            arquivo.write(f"URL: {trecho['url']}\n")
            arquivo.write(f"Similaridade: {trecho['similaridade']:.6f}\n")
            arquivo.write(f"Nota hibrida: {trecho['nota_hibrida']:.6f}\n")
            arquivo.write(f"Trecho: {trecho['texto']}\n")

    print("\nRelatorio atualizado:", ARQUIVO_RELATORIO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RAG sobre as notícias do ge.globo."
    )
    parser.add_argument(
        "--teste",
        action="store_true",
        help="roda as perguntas de teste fixas"
    )
    parser.add_argument(
        "--avaliar",
        action="store_true",
        help="avalia a busca com o gabarito"
    )
    args = parser.parse_args()

    if args.avaliar:
        avaliar_busca()
        raise SystemExit

    limiar, amostras, sobreposicao = calibrar_limiar()

    print("\nCalibração do limiar:")
    for texto, relevante, score in amostras:
        print(f"  {score:.3f} | esperado={relevante!s:5} | {texto}")
    print(f"Limiar: {limiar:.3f}")
    if sobreposicao:
        print("ATENÇÃO: as classes se sobrepõem; nenhum limiar separa todos os exemplos.")

    salvar_calibracao(limiar, amostras, sobreposicao)

    if args.teste:
        for pergunta in PERGUNTAS_TESTE:
            resultado = perguntar_rag(pergunta, limiar)
            mostrar_resultado(resultado)
            salvar_relatorio(resultado, limiar)
    else:
        while True:
            try:
                pergunta = input("\nPergunta (ou 'sair'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nPrograma encerrado.")
                break

            if pergunta.casefold() == "sair":
                print("Programa encerrado.")
                break

            if not pergunta:
                continue

            resultado = perguntar_rag(pergunta, limiar)
            mostrar_resultado(resultado)
            salvar_relatorio(resultado, limiar)
