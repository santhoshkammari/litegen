
import os

from openai import OpenAI
from typing import Optional, List, Dict, Literal
from ollama._utils import convert_function_to_tool
from litegen._types import ModelType


class LLM:
    def __init__(
        self,
        api_key: str = 'ollama',
        base_url: str = None,
        debug: bool = None
    ):
        self.debug: bool | None = debug
        self.api_key = self._get_api_key(api_key)
        self.base_url = self._get_base_url(self.api_key, base_url)
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        self.DEFAULT_MODELS: Dict = {
            "ollama": "qwen2.5:0.5b-instruct",
            "dsollama": "qwen2.5:7b-instruct",
            "huggingchat": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "huggingchat_nemo": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            "huggingchat_hermes": "NousResearch/Hermes-3-Llama-3.1-8B",
            "huggingchat_phimini": "microsoft/Phi-3.5-mini-instruct",
        }

    @staticmethod
    def build_messages(
        system_prompt: str = None,
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
        model: str = None,
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
        """Create a chat completion with either messages or individual components."""
        # If messages not provided, build from components
        if model is None:
            model = self._get_model_name()
        if messages is None:
            messages = self.build_messages(system_prompt, prompt, context)
        elif isinstance(messages, str):
            messages = self.handle_str_messages(messages, system_prompt)

        # Prepare Tools for Calling
        if tools:
            tools = self._prepare_tools(tools)

        if self.debug:
            print('-'*50)
            print(f'{self.api_key=}')
            print(f'{self.base_url=}')
            print(f'{model=}')
            print(f'{messages=}')
            print(f'{tools=}')
            print('-'*50)

        # Get response from API
        if kwargs.get("response_format", None) is None:
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
        else:
            response = self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
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
        _messages = []
        if system_prompt:
            _messages.append({"role": "system", "content": system_prompt})

        _messages.append({"role": "user", "content": messages})

        return _messages

    def _get_base_url(self, api_key, base_url):
        if os.environ.get('OPENAI_BASE_URL', None): return os.environ['OPENAI_BASE_URL']
        if base_url: return base_url

        if 'huggingchat' in api_key:
            _api_key = "huggingchat"  # huggingchat_code etc..
        else:
            _api_key = api_key

        match _api_key:
            case 'dsollama':
                return "http://192.168.170.76:11434/v1"
            case 'ollama':
                return "http://localhost:11434/v1"
            case 'huggingchat':
                return "http://localhost:11437/v1"
            case 'llamacpp':
                return "http://localhost:11438/v1"
            case _:
                return None

    def _get_api_key(self, api_key: str | None = None) -> str:
        _env_api_key = os.environ.get('OPENAI_API_KEY')
        if _env_api_key:
            return _env_api_key
        return api_key

    def _prepare_tools(self, tools):
        tools = [t if isinstance(t, dict) else convert_function_to_tool(t) for t in tools]
        return tools

    def __call__(
        self,
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

        response_format = kwargs.pop("response_format", None)
        kwargs['response_format'] = response_format

        res = self.completion(
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
        self,
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
        res = self.completion(
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

    def _get_model_name(self):
        model = os.environ.get('OPENAI_MODEL_NAME')
        if model is None:
            model = self.DEFAULT_MODELS.get(self.api_key, None)
            if model is None:
                raise ValueError("Missing required environment variable: OPENAI_MODEL_NAME or pass model")
        return model


if __name__ == '__main__':
    llm = LLM()
    print(llm("hai"))
