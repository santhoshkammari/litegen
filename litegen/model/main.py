#ailite/model/main.py
from typing import Optional, Dict, List
from litegen.model._oai import OmniLLMClient
from litegen.model._serving_models import ServingModel

__client = None
genai = ServingModel()

def get_client():
    global __client
    if __client is None:
        __client = OmniLLMClient()
    return __client

def lazy_completion(
    model: str,
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


