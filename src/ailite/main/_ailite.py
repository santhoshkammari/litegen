from typing import List, Dict, Union, Generator, Optional

from ..main._model._api.types._model_types import MODELS_TYPE


def stream_response(response):
    for chunk in response.iter_content(decode_unicode=True):
        if chunk:
            yield chunk

class ClaudeEngine:
    def __init__(
        self,
        model: MODELS_TYPE = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        api_url: str = "http://localhost:11435"
    ):
        self.model = model
        self.api_url = api_url.rstrip('/')

    def __call__(
        self,
        messages: Union[List[Dict[str, str]], str],
        model:MODELS_TYPE = None,
        conversation: str = False,
        websearch: bool = False,
        stream: bool = False,
        json: bool = False
    ) -> str|Generator:
        """
        Send a request to the API and get the response.

        Args:
            messages: Either a string prompt or a list of message dictionaries

        Returns:
            String response from the API
        """
        payload = {
                "prompt": messages,
                "model": model if model is not None else self.model,
                "conversation": conversation,
                "stream": stream,
                "websearch": websearch,
            }
        response = requests.post(
            f"{self.api_url}/v1/generate",
            json=payload,
            stream=stream
        )
        response.raise_for_status()
        if stream:
            return stream_response(response)
        else:
            if json:
                return response.json()
            return response.json()['message']['content']


from typing import List, Dict, Union
import requests


def ai(
    prompt_or_messages: Union[List[Dict[str, str]], str],
    model: MODELS_TYPE = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    api_url: str = "http://localhost:11435",
    conversation: bool = False,
    websearch: bool = False,
    stream: bool = False,
    json:bool = False,
    assistant:Optional[str] = None
) -> str|Generator:
    """
    Send a request to the AI API and get the response.

    Args:
        messages: Either a string prompt or a list of message dictionaries
        model: The AI model to use for generation
        api_url: The base URL for the API endpoint
        conversation: Whether to maintain conversation context
        websearch: Whether to enable web search capability
        stream: Whether to stream the response

    Returns:
        String response from the API

    Raises:
        requests.exceptions.RequestException: If the API request fails
    """
    # Clean up the API URL by removing trailing slashes
    api_url = api_url.rstrip('/')

    # Prepare the request payload
    payload = {
        "prompt": prompt_or_messages,
        "model": model,
        "conversation": conversation,
        "stream": stream,
        "websearch": websearch,
        "assistant":assistant
    }

    # Send the request and get response
    response = requests.post(
        f"{api_url}/v1/generate",
        json=payload,
        stream=stream
    )
    response.raise_for_status()
    if stream:
        return stream_response(response)
    else:
        if json:
            return response.json()
        return response.json()['message']['content']