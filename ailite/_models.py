from langchain_ollama import ChatOllama


def qwen7b_gpu():
    return ChatOllama(base_url="http://192.168.170.76:11434",model="qwen2.5:7b-instruct")

def qwen2p5_local():
    return ChatOllama(model="qwen2.5:0.5b-instruct")