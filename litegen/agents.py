from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Sequence, Callable
from abc import ABC, abstractmethod
import asyncio
import os

from openai import OpenAI
from openai.types.chat import ChatCompletion
from ._agent_utils import convert_function_to_tool


@dataclass
class ModelInfo:
    """Model capability information"""
    vision: bool = False
    json_output: bool = False
    function_calling: bool = True
    family: str = "none"


@dataclass
class AgentResponse:
    """Response from an agent"""
    content: str
    raw_response: Optional[ChatCompletion] = None
    usage: Optional[Dict] = None


class ModelClient:
    """Base model client wrapper for OpenAI"""

    BASE_URLS = {
        'ollama': 'http://localhost:11434/v1',
        'dsollama': 'http://192.168.170.76:11434/v1',
        'huggingchat': 'http://localhost:11437/v1'
    }

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_info: Optional[ModelInfo] = None,
        **kwargs
    ):
        self.model = model or os.getenv('OPENAI_MODEL_NAME')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', 'ollama')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL') or self.BASE_URLS.get(self.api_key)
        self.model_info = model_info or ModelInfo()
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, **kwargs)

    async def create(self, messages: List[Dict], **kwargs) -> ChatCompletion:
        """Create a chat completion using OpenAI's format"""
        if tools := kwargs.get('tools'):
            kwargs["tools"] = [
                convert_function_to_tool(f) if callable(f) else f
                for f in tools
            ]

        return await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            **kwargs
        )


class BaseAgent(ABC):
    """Base agent class that defines the interface"""

    def __init__(
        self,
        model_client: ModelClient,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.name = name
        self.client = model_client
        self.conversation_history: List[Dict] = []
        self.registered_tools: List[Any] = []

        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })

    @abstractmethod
    async def run(self, message: str, **kwargs) -> AgentResponse:
        """Run the agent with a message"""
        pass

    def __call__(self, message: str, **kwargs) -> AgentResponse:
        """Synchronous wrapper for run()"""
        try:
            loop = asyncio.get_running_loop()
            return self.run(message, **kwargs) if loop.is_running() else loop.run_until_complete(
                self.run(message, **kwargs))
        except RuntimeError:
            return asyncio.run(self.run(message, **kwargs))

    def register_tool(self, func: Union[Callable, List[Callable]]) -> 'BaseAgent':
        """Register one or more tools for use in the agent"""
        if isinstance(func, list):
            self.registered_tools.extend(
                convert_function_to_tool(f) if callable(f) else f
                for f in func
            )
        else:
            self.registered_tools.append(
                convert_function_to_tool(func) if callable(func) else func
            )
        return self

    # Alias for register_tool
    tool = register_tool


class Agent(BaseAgent):
    """Main agent implementation"""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_client: Optional[ModelClient] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            model_client=model_client or ModelClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            ),
            system_prompt=system_prompt,
        )

    async def run(self, message: str, **kwargs) -> AgentResponse:
        """Run the agent with a message"""
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        tools = [*kwargs.pop('tools', []), *self.registered_tools]
        response = await self.client.create(
            messages=self.conversation_history,
            tools=tools,
            **kwargs
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return AgentResponse(
            content=assistant_message,
            raw_response=response,
            usage=response.usage.model_dump() if response.usage else None
        )
