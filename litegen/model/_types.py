from typing import Literal

ModelType = Literal[
    "smollm2:1.7b-instruct-fp16",
    "smollm2:135m-instruct-fp16",
    "smollm2:135m-instruct-q4_K_M",
    "qwen2.5-coder:1.5b-instruct",
    "qwen2.5-coder:0.5b-instruct",
    "qwen2.5:3b-instruct",
    "qwen2.5:0.5b-instruct",
    "qwen2.5:1.5b-instruct",
    "qwen2.5:7b-instruct",
    "qwen2.5-coder:3b-instruct-q4_k_m",
    "qwen2.5-coder:7b-instruct",
    "qwen2.5-coder:14b-instruct-q4_K_M",
    "qwen2.5-coder:32b-instruct-q4_K_M",
    "llama3.2:1b-instruct-q4_K_M",
    "llama3.2:1b-instruct-fp16",
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-fp16",
    "gpt-4o-mini"
]
