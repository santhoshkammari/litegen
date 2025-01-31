# ailite/model/main.py
from typing import Optional, Dict, List
from litegen._oai import BaseOmniLLMClient
from litegen._types import ModelType

__client = None
__api_key = None
__base_url = None


def get_client():
    global __client, __api_key, __base_url
    if __client is None or __client.api_key != __api_key or __client.base_url != __base_url:
        __client = BaseOmniLLMClient()
        __api_key = __client.api_key
        __base_url = __client.base_url
    return __client


def lazy_completion(
    messages: Optional[List[Dict[str, str]]] | str = None,
    model: ModelType = None,
    system_prompt: str = None,
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
    system_prompt: str = None,
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

    response_format = kwargs.pop("response_format",None)

    kwargs['response_format']=response_format

    res = client.completion(
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
    if response_format is None:
        return res.choices[0].message.content
    else:
        return res.choices[0].message.parsed


def print_stream_completion(
    model: ModelType,
    messages: Optional[List[Dict[str, str]]] | str = None,
    system_prompt: str = None,
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


if __name__ == '__main__':
    import os

    from pydantic import BaseModel

    os.environ['OPENAI_API_KEY'] = "dsollama"
    os.environ['OPENAI_MODEL_NAME'] = "qwen2.5:7b-instruct"
    from litegen import genai


    class Friend(BaseModel):
        name: str
        age: int


    print(genai('hi,my friend name is santhosh and age is 23', response_format=Friend))
