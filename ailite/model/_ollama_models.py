from langchain_ollama import ChatOllama

def __get_base_url(gpu=False):
    if gpu:
        return "http://192.168.170.76:11434"
    else:
        return "http://localhost:11434"


def qwen2p5_7b(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="qwen2.5:7b-instruct")


def qwen2p5_local(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="qwen2.5:0.5b-instruct")


def llama3_2_3b_instruct_q4_K_M(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="llama3.2:3b-instruct-q4_K_M")


def qwen2_5_coder_1_5b_instruct(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="qwen2.5-coder:1.5b-instruct")


def qwen2_5_coder_0_5b_instruct(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="qwen2.5-coder:0.5b-instruct")


def llama3_2_latest(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="llama3.2:latest")


def qwen2_5_3b_instruct(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="qwen2.5:3b-instruct")


def qwen2_5_0_5b_instruct(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="qwen2.5:0.5b-instruct")


def nomic_embed_text_latest(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="nomic-embed-text:latest")


def llama3_2_1b_instruct_q4_K_M(gpu=False):
    return ChatOllama(base_url=__get_base_url(gpu), model="llama3.2:1b-instruct-q4_K_M")
