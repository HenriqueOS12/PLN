# python -m pip install transformers gensim scikit-learn nltk pandas langgraph
# Com GPU (ex.: Google Colab): python -m pip install accelerate bitsandbytes
#
# Uso:
#   python scripts/chatbot_rag_ge.py              # memória manual (lista Python)
#   python scripts/chatbot_rag_ge.py --langgraph  # memória com LangGraph

import argparse
import ast
import re
import uuid
from datetime import datetime
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch

from gensim.models import KeyedVectors, Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PASTA_RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_ENTRADA = PASTA_RAIZ / "dados" / "processados" / "noticias_ge_processado.csv"
PASTA_SAIDA = PASTA_RAIZ / "resultados" / "chatbot"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Se a base de notícias mudar, apague este arquivo para treinar o Word2Vec de novo.
ARQUIVO_WORD2VEC = PASTA_SAIDA / "vetores_word2vec.kv"
ARQUIVO_RELATORIO = PASTA_SAIDA / "relatorio_chatbot.txt"

RANDOM_STATE = 42

MODELO_GPU = "Qwen/Qwen2.5-7B-Instruct"
MODELO_CPU = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_TOKENS_RESPOSTA = 300
TEMPERATURA = 0.3
TOP_P = 0.9
MAX_TURNOS = 4

QUANTIDADE_NOTICIAS = 3
MAX_CARACTERES_NOTICIA = 1200

PESO_TFIDF_HIBRIDO = 0.5
TAMANHO_VETOR_W2V = 100

torch.manual_seed(RANDOM_STATE)


for caminho_nltk, recurso in [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]:
    try:
        nltk.data.find(caminho_nltk)
    except LookupError:
        nltk.download(recurso, quiet=True)

stopwords_pt = set(stopwords.words("portuguese"))

df = pd.read_csv(
    ARQUIVO_ENTRADA,
    encoding="utf-8-sig"
).fillna("")

df["tokens_sem_stopwords"] = df[
    "tokens_sem_stopwords"
].apply(ast.literal_eval)

datas = pd.to_datetime(df["data_publicacao"], dayfirst=True, errors="coerce")
PERIODO = f"{datas.min():%d/%m/%Y} e {datas.max():%d/%m/%Y}"


def tokens_bow(texto):

    tokens = word_tokenize(texto, language="portuguese")

    return [
        token.casefold()
        for token in tokens
        if token.isalpha() and token.casefold() not in stopwords_pt
    ]


def tokens_word2vec(texto):

    texto = re.sub(r"[^a-zA-ZÀ-ÿ0-9]+", " ", texto.casefold())

    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", texto)


def carregar_word2vec():

    if ARQUIVO_WORD2VEC.exists():
        return KeyedVectors.load(str(ARQUIVO_WORD2VEC))

    print("Treinando Word2Vec (só na primeira execução)...")

    modelo = Word2Vec(
        sentences=[t for t in df["tokens_sem_stopwords"] if t],
        vector_size=TAMANHO_VETOR_W2V,
        window=5,
        min_count=1,
        sg=1,
        seed=RANDOM_STATE,
        workers=1,
        epochs=100
    )

    modelo.wv.save(str(ARQUIVO_WORD2VEC))

    return modelo.wv


def media_word2vec(tokens):
    vetores = [vetores_w2v[t] for t in tokens if t in vetores_w2v]

    if not vetores:
        return np.zeros(TAMANHO_VETOR_W2V, dtype=np.float32)

    return np.mean(vetores, axis=0)


def escala_0_1(valores):

    minimo, maximo = valores.min(), valores.max()

    if maximo == minimo:
        return np.zeros_like(valores)

    return (valores - minimo) / (maximo - minimo)


tfidf_vectorizer = TfidfVectorizer(
    analyzer=lambda tokens: tokens
)

matriz_tfidf = tfidf_vectorizer.fit_transform(
    df["tokens_sem_stopwords"]
)

vetores_w2v = carregar_word2vec()

embeddings_w2v = normalize(np.asarray(
    [media_word2vec(tokens) for tokens in df["tokens_sem_stopwords"]],
    dtype=np.float32
))


def buscar_noticias(consulta, quantidade=QUANTIDADE_NOTICIAS):

    vetor_tfidf = normalize(tfidf_vectorizer.transform([tokens_bow(consulta)]))
    vetor_w2v = media_word2vec(tokens_word2vec(consulta))

    notas_tfidf = (
        (matriz_tfidf @ vetor_tfidf.T).toarray().ravel()
        if vetor_tfidf.nnz else None
    )
    notas_w2v = (
        embeddings_w2v @ normalize([vetor_w2v])[0]
        if vetor_w2v.any() else None
    )

    if notas_tfidf is None and notas_w2v is None:
        return []

    if notas_w2v is None:
        notas = notas_tfidf
    elif notas_tfidf is None:
        notas = notas_w2v
    else:
        notas = (
            PESO_TFIDF_HIBRIDO * escala_0_1(notas_tfidf)
            + (1 - PESO_TFIDF_HIBRIDO) * escala_0_1(notas_w2v)
        )

    melhores = np.argsort(notas)[::-1][:quantidade]

    return [
        {
            "titulo": df["titulo"].iloc[indice],
            "data": df["data_publicacao"].iloc[indice],
            "url": df["url"].iloc[indice],
            "texto": df["texto"].iloc[indice],
            "nota": float(notas[indice]),
        }
        for indice in melhores
    ]


def consulta_para_busca(perguntas):

    return " ".join(perguntas[-2:])


PROMPT_SISTEMA = f"""
Você é um assistente que responde perguntas sobre notícias de futebol
publicadas no portal ge.globo entre {PERIODO}.

Regras:
- Responda em português brasileiro, de forma clara e objetiva.
- Use apenas as informações das notícias fornecidas e da conversa anterior.
- Se a informação não estiver nas notícias, diga que não a encontrou na base.
- Ao usar uma notícia, cite o número dela entre colchetes, por exemplo [1].
""".strip()


def mensagem_com_noticias(pergunta, noticias):

    if not noticias:
        return (
            "Nenhuma notícia da base foi encontrada para esta pergunta.\n\n"
            f"Pergunta: {pergunta}"
        )

    blocos = []

    for numero, noticia in enumerate(noticias, start=1):
        texto = " ".join(noticia["texto"].split())

        if len(texto) > MAX_CARACTERES_NOTICIA:
            texto = texto[:MAX_CARACTERES_NOTICIA] + "..."

        blocos.append(
            f"[{numero}] {noticia['titulo']} ({noticia['data']})\n{texto}"
        )

    return (
        "Notícias encontradas na base:\n\n"
        + "\n\n".join(blocos)
        + f"\n\nPergunta: {pergunta}"
    )


def mostrar_fontes(noticias):
    if not noticias:
        print("(nenhuma notícia encontrada na base)")
        return

    print("Fontes consultadas:")

    for numero, noticia in enumerate(noticias, start=1):
        print(f"  [{numero}] {noticia['titulo']}")
        print(f"      {noticia['url']}")


if torch.cuda.is_available():

    NOME_MODELO = MODELO_GPU

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        NOME_MODELO,
        quantization_config=quantization_config,
        device_map="auto"
    )

else:

    NOME_MODELO = MODELO_CPU

    model = AutoModelForCausalLM.from_pretrained(
        NOME_MODELO,
        dtype=torch.float32
    )

tokenizer = AutoTokenizer.from_pretrained(NOME_MODELO)

DESCRICAO_MODELO = f"{NOME_MODELO} ({'GPU, 4 bits' if torch.cuda.is_available() else 'CPU'})"

print(f"Modelo carregado: {DESCRICAO_MODELO}")


def gerar_resposta(messages, max_tokens=MAX_TOKENS_RESPOSTA):

    texto = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        texto,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=TEMPERATURA,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )

    novos_tokens = outputs[0][inputs.input_ids.shape[1]:]

    resposta = tokenizer.decode(
        novos_tokens,
        skip_special_tokens=True
    )

    return resposta.strip()


def contar_tokens(messages):

    texto = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return tokenizer(texto, return_tensors="pt").input_ids.shape[1]


def salvar_relatorio(pergunta, resposta, noticias, memoria):

    with open(
        ARQUIVO_RELATORIO,
        "a",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write("\n" + "=" * 80 + "\n")
        arquivo.write(f"PERGUNTA: {pergunta}\n")
        arquivo.write("=" * 80 + "\n")
        arquivo.write(f"Data: {datetime.now():%d/%m/%Y %H:%M:%S}\n")
        arquivo.write(f"Modelo: {DESCRICAO_MODELO}\n")
        arquivo.write(f"Memoria: {memoria}\n")

        arquivo.write("\n========== RESPOSTA ==========\n")
        arquivo.write(f"{resposta}\n")

        arquivo.write("\n========== NOTICIAS USADAS ==========\n")

        if not noticias:
            arquivo.write("Nenhuma noticia encontrada.\n")

        for posicao, noticia in enumerate(noticias, start=1):
            arquivo.write(f"\nPosicao: {posicao}\n")
            arquivo.write(f"Titulo: {noticia['titulo']}\n")
            arquivo.write(f"Data: {noticia['data']}\n")
            arquivo.write(f"URL: {noticia['url']}\n")
            arquivo.write(f"Nota da busca: {noticia['nota']:.6f}\n")

    print("\nRelatorio atualizado:", ARQUIVO_RELATORIO)


historico = [
    {
        "role": "system",
        "content": PROMPT_SISTEMA
    }
]

ultimo_contexto = []


def montar_contexto(max_turnos=MAX_TURNOS):

    sistema = historico[0]

    max_mensagens_recentes = 2 * max_turnos - 1

    recentes = historico[1:][-max_mensagens_recentes:]

    return [sistema] + recentes


def conversar_com_janela(mensagem, max_tokens=MAX_TOKENS_RESPOSTA, max_turnos=MAX_TURNOS):

    global ultimo_contexto

    perguntas = [m["content"] for m in historico if m["role"] == "user"]
    noticias = buscar_noticias(consulta_para_busca(perguntas + [mensagem]))

    historico.append(
        {
            "role": "user",
            "content": mensagem
        }
    )

    contexto = montar_contexto(max_turnos=max_turnos)
    contexto[-1] = {
        "role": "user",
        "content": mensagem_com_noticias(mensagem, noticias)
    }

    resposta = gerar_resposta(contexto, max_tokens=max_tokens)

    historico.append(
        {
            "role": "assistant",
            "content": resposta
        }
    )

    ultimo_contexto = contexto

    return resposta, noticias


def mostrar_historico():

    for numero, mensagem in enumerate(historico):
        print(f"[{numero}] {mensagem['role'].upper()}")
        print(mensagem["content"])
        print("-" * 60)


def limpar_memoria():

    historico.clear()

    historico.append(
        {
            "role": "system",
            "content": PROMPT_SISTEMA
        }
    )

    print("Memória da conversa apagada.")


def executar_chat_manual():

    print("\nChatbot de notícias do ge.globo (memória manual).")
    print("Comandos: /sair | /limpar | /memoria | /tokens")

    while True:

        try:
            mensagem = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConversa encerrada.")
            break

        comando = mensagem.lower()

        if comando == "/sair":
            print("Conversa encerrada.")
            break

        if comando == "/limpar":
            limpar_memoria()
            continue

        if comando == "/memoria":
            mostrar_historico()
            continue

        if comando == "/tokens":
            print("Tokens no histórico completo:", contar_tokens(historico))
            if ultimo_contexto:
                print("Tokens enviados na última pergunta (janela + notícias):",
                      contar_tokens(ultimo_contexto))
            continue

        if not mensagem:
            continue

        resposta, noticias = conversar_com_janela(mensagem)

        print("\nChatbot:")
        print(resposta)
        print()
        mostrar_fontes(noticias)

        salvar_relatorio(mensagem, resposta, noticias, "manual")


def criar_chatbot_langgraph():

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    class EstadoChat(MessagesState):
        noticias: list

    def recuperar(state: EstadoChat):
        perguntas = [
            m.content for m in state["messages"]
            if isinstance(m, HumanMessage)
        ]

        return {"noticias": buscar_noticias(consulta_para_busca(perguntas))}

    def responder_qwen(state: EstadoChat):
        mensagens = []

        for mensagem in state["messages"]:
            if isinstance(mensagem, HumanMessage):
                mensagens.append({"role": "user", "content": mensagem.content})
            elif isinstance(mensagem, AIMessage):
                mensagens.append({"role": "assistant", "content": mensagem.content})

        recentes = mensagens[-(2 * MAX_TURNOS - 1):]
        recentes[-1] = {
            "role": "user",
            "content": mensagem_com_noticias(recentes[-1]["content"], state["noticias"])
        }

        resposta = gerar_resposta(
            [{"role": "system", "content": PROMPT_SISTEMA}] + recentes
        )

        return {"messages": [AIMessage(content=resposta)]}

    fluxo = StateGraph(EstadoChat)
    fluxo.add_node("recuperar", recuperar)
    fluxo.add_node("qwen", responder_qwen)
    fluxo.add_edge(START, "recuperar")
    fluxo.add_edge("recuperar", "qwen")
    fluxo.add_edge("qwen", END)

    return fluxo.compile(checkpointer=InMemorySaver())


def executar_chat_langgraph():

    from langchain_core.messages import HumanMessage

    chatbot = criar_chatbot_langgraph()

    sessao = str(uuid.uuid4())

    print("\nChatbot de notícias do ge.globo (memória com LangGraph).")
    print("Comandos: /sair | /nova | /memoria")
    print("Sessão:", sessao)

    while True:

        try:
            mensagem = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConversa encerrada.")
            break

        configuracao = {"configurable": {"thread_id": sessao}}
        comando = mensagem.lower()

        if comando == "/sair":
            print("Conversa encerrada.")
            break

        if comando == "/nova":
            sessao = str(uuid.uuid4())
            print("Nova sessão:", sessao)
            continue

        if comando == "/memoria":
            estado = chatbot.get_state(configuracao)
            mensagens = estado.values.get("messages", [])
            for indice, item in enumerate(mensagens, start=1):
                print(f"{indice}. {item.type}: {item.content}")
            print(f"Total de mensagens: {len(mensagens)}")
            continue

        if not mensagem:
            continue

        resultado = chatbot.invoke(
            {"messages": [HumanMessage(content=mensagem)]},
            config=configuracao,
        )

        resposta = resultado["messages"][-1].content

        print("\nChatbot:")
        print(resposta)
        print()
        mostrar_fontes(resultado["noticias"])

        salvar_relatorio(mensagem, resposta, resultado["noticias"], f"LangGraph (sessao {sessao})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Chatbot com RAG sobre as notícias do ge.globo."
    )
    parser.add_argument(
        "--langgraph",
        action="store_true",
        help="usa a memória com LangGraph em vez da lista Python"
    )

    if parser.parse_args().langgraph:
        executar_chat_langgraph()
    else:
        executar_chat_manual()
