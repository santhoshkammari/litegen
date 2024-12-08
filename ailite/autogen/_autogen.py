import os
from typing import Any, Dict, List, Optional, Sequence, Union, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, ConfigDict
from autogen_core.components.models import (
    ModelCapabilities,
    RequestUsage,
    LLMMessage,
    CreateResult,
    ChatCompletionTokenLogprob,
    ChatCompletionClient,
)
from autogen_core.components.tools import Tool, ToolSchema
from autogen_core.base import CancellationToken

# from dotenv import load_dotenv
# load_dotenv("../../.env")


# HF_API_KEY = os.environ.get("HF_API_KEY")


class AILiteConfig(BaseModel):
    model: str = "Qwen/Qwen2.5-72B-Instruct"
    temperature: float = 0.7
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    api_key: str = "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"
    base_url: str = "https://api-inference.huggingface.co/v1/"

    model_config = ConfigDict(protected_namespaces=())

class OpenAIChatCompletionClient(ChatCompletionClient):
    """OpenAI chat models for HuggingFace."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = 0.7,
        kwargs: Dict[str, Any] = None,
        api_key: str = "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",
        base_url: str = "https://api-inference.huggingface.co/v1/",
    ):
        self.config = AILiteConfig(
            model=model,
            temperature=temperature,
            kwargs=kwargs or {},
            api_key=api_key,
            base_url=base_url,
        )
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[Dict[str, str]],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        """Create a completion."""
        response = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **{**self.config.kwargs, **extra_create_args}
        )

        usage = RequestUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        return CreateResult(
            content=response.choices[0].message.content or "",
            finish_reason=response.choices[0].finish_reason,
            usage=usage,
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[Dict[str, str]],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """Create a streaming completion."""
        stream = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **{**self.config.kwargs, **extra_create_args}
        )

        content_parts = []
        finish_reason = None
        usage = None

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                content_parts.append(delta.content)
                yield delta.content

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            if chunk.usage:
                usage = RequestUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                )

        if not usage:
            usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        result = CreateResult(
            content="".join(content_parts),
            finish_reason=finish_reason,
            usage=usage,
            cached=False,
        )
        yield result

    def actual_usage(self) -> RequestUsage:
        """Get the actual usage."""
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        """Get the total usage."""
        return self._total_usage

    @property
    def capabilities(self) -> ModelCapabilities:
        """Get the capabilities."""
        return ModelCapabilities(
            completion=True,
            chat=True,
            function_calling=False,
            vision=False,
            json_output=False,
        )