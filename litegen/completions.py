# ailite/model/main.py
from typing import Optional, Dict, List
from litegen._oai import OmniLLMClient
from litegen._types import ModelType

__client = None
__api_key = None
__base_url = None


def get_client():
    global __client, __api_key, __base_url
    if __client is None or __client.api_key != __api_key or __client.base_url != __base_url:
        __client = OmniLLMClient()
        __api_key = __client.api_key
        __base_url = __client.base_url
    return __client


def lazy_completion(
    messages: Optional[List[Dict[str, str]]] | str = None,
    model: ModelType = None,
    system_prompt: str = "You are helpful Assistant",
    prompt: str = "",
    context: Optional[List[Dict[str, str]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    stop: Optional[List[str]] = None,
    tools=None,
    **kwargs
):
    client = get_client()
    return client.completion(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        prompt=prompt,
        context=context,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        stop=stop,
        tools=tools,
        **kwargs
    )


def genai(
    messages: Optional[List[Dict[str, str]]] | str = None,
    model: ModelType = None,
    system_prompt: str = "You are helpful Assistant",
    prompt: str = "",
    context: Optional[List[Dict[str, str]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    stop: Optional[List[str]] = None,
    tools=None,
    **kwargs
):
    client = get_client()
    return client.completion(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        prompt=prompt,
        context=context,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        stop=stop,
        tools=tools,
        **kwargs
    ).choices[0].message.content


def print_stream_completion(
    model: ModelType,
    messages: Optional[List[Dict[str, str]]] | str = None,
    system_prompt: str = "You are helpful Assistant",
    prompt: str = "",
    context: Optional[List[Dict[str, str]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    stop: Optional[List[str]] = None,
    tools=None,
    **kwargs
):
    client = get_client()
    res = client.completion(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        prompt=prompt,
        context=context,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        stop=stop,
        tools=tools,
        **kwargs
    )
    for x in res:
        print(x.choices[0].delta.content, end="", flush=True)
