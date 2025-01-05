# ailite/model/_ollama_models.py

import os

from openai import OpenAI
from typing import Optional, List, Dict, Literal
from langchain_core.utils.function_calling import convert_to_openai_function



class OmniLLMClient:
    def __init__(
        self,
        api_key: str = 'ollama',
        base_url: str = "http://localhost:11434/v1",
        gpu: bool = False
    ):
        self.base_url = self._get_base_url(api_key, base_url, gpu)
        self.api_key = self._get_api_key(api_key)
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    @staticmethod
    def build_messages(
        system_prompt: str = "",
        prompt: str = "",
        context: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Build messages list from components."""
        _messages = []

        # Add system prompt if provided
        if system_prompt:
            _messages.append({"role": "system", "content": system_prompt})

        # Add context (previous conversation) if provided
        if context:
            _messages.extend(context)

        # Add current prompt if provided
        if prompt:
            _messages.append({"role": "user", "content": prompt})

        return _messages

    def completion(
        self,
        messages: Optional[List[Dict[str, str]]] | str = None,
        model: str=None,
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
        """Create a chat completion with either messages or individual components."""
        # If messages not provided, build from components
        if messages is None:
            messages = self.build_messages(system_prompt, prompt, context)
        elif isinstance(messages, str):
            messages = self.handle_str_messages(messages, system_prompt)

        #Prepare Tools for Calling
        if tools:
            tools = self._prepare_tools(tools)

        # Get response from API
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop,
            tools=tools,
            **kwargs
        )
        return response

    def _stream_to_string(self, stream_response):
        """Convert streaming response to string iterator."""
        for chunk in stream_response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _extract_response_text(self, response) -> str:
        """Extract text from non-streaming response."""
        return response.choices[0].message.content or ""

    @staticmethod
    def handle_str_messages(messages, system_prompt):
        """Handle string messages and build them into a list of messages."""
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": messages}]

    def _get_base_url(self, api_key, base_url, gpu):
        match (gpu, api_key):
            case (True, _):
                return "http://192.168.170.76:11434/v1"
            case (False, 'ollama'):
                return "http://localhost:11434/v1"
            case _:
                return None

    def _get_api_key(self, api_key: str) -> str:
        _env_api_key = os.environ.get('OPENAI_API_KEY')
        if _env_api_key:
            return _env_api_key
        return api_key

    def _prepare_tools(self, tools):
        return [t if isinstance(t,dict) else convert_to_openai_function(t) for t in tools]

