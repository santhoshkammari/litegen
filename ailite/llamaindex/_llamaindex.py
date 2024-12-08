import os
from typing import Any, Dict, List, Optional, Sequence, Union, cast, Literal

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.callbacks import CallbackManager
from openai import OpenAI as BaseOpenAI, AsyncOpenAI, NOT_GIVEN, NotGiven

from ailitellm import HFModelType

from pydantic import Field

from dotenv import load_dotenv
load_dotenv("../../.env")
HF_API_KEY = os.environ.get("HF_API_KEY")


class OpenAI(FunctionCallingLLM):
    model: str = Field(default="Qwen/Qwen2.5-72B-Instruct")
    temperature: float = Field(default=0.0)
    max_tokens: Optional[int] = Field(default=4028)
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int = Field(default=3)
    timeout: float = Field(default=60.0)
    api_key: str = Field(default="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")
    base_url: str = Field(default="https://api-inference.huggingface.co/v1/")
    reuse_client: bool = Field(default=True)

    def __init__(
        self,
        model: HFModelType = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
            **kwargs
        )
        self._client = BaseOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=4096,  # Assuming default context window
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint."""
        messages = [{"role": "user", "content": prompt}]
        chat_response = self._client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **{**self.additional_kwargs, **kwargs}
        )
        return CompletionResponse(text=chat_response.choices[0].message.content or "")

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async completion endpoint."""
        messages = [{"role": "user", "content": prompt}]
        chat_response = await self._aclient.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **{**self.additional_kwargs, **kwargs}
        )
        return CompletionResponse(text=chat_response.choices[0].message.content or "")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Streaming completion endpoint."""
        messages = [{"role": "user", "content": prompt}]
        chat_response = self._client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **{**self.additional_kwargs, **kwargs}
        )

        def gen() -> CompletionResponseGen:
            text = ""
            for chunk in chat_response:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        text += delta
                        yield CompletionResponse(text=text, delta=delta)

        return gen()

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint."""
        messages = [{"role": "user", "content": prompt}]
        chat_response = await self._aclient.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **{**self.additional_kwargs, **kwargs}
        )

        async def gen() -> CompletionResponseAsyncGen:
            text = ""
            async for chunk in chat_response:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        text += delta
                        yield CompletionResponse(text=text, delta=delta)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint."""
        openai_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
        response = self._client.chat.completions.create(
            messages=openai_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **{**self.additional_kwargs, **kwargs}
        )

        msg_content = response.choices[0].message.content or ""
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=msg_content))

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async chat endpoint."""
        openai_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
        response = await self._aclient.chat.completions.create(
            messages=openai_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **{**self.additional_kwargs, **kwargs}
        )

        msg_content = response.choices[0].message.content or ""
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=msg_content))

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Streaming chat endpoint."""
        openai_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
        chat_response = self._client.chat.completions.create(
            messages=openai_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **{**self.additional_kwargs, **kwargs}
        )

        def gen() -> ChatResponseGen:
            content = ""
            for chunk in chat_response:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        content += delta
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=content
                            ),
                            delta=delta
                        )

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint."""
        openai_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
        chat_response = await self._aclient.chat.completions.create(
            messages=openai_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **{**self.additional_kwargs, **kwargs}
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            async for chunk in chat_response:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        content += delta
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=content
                            ),
                            delta=delta
                        )

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare chat with tools."""
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": [tool.metadata.to_openai_tool() for tool in tools],
            **kwargs,
        }

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any
    ) -> List[ToolSelection]:
        """Get tool calls from response."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if not tool_calls and error_on_no_tool_call:
            raise ValueError("No tool calls found in response")

        selections = []
        for tool_call in tool_calls:
            selections.append(
                ToolSelection(
                    tool_name=tool_call.function.name,
                    tool_id=tool_call.id,
                    tool_kwargs=tool_call.function.arguments
                )
            )
        return selections

if __name__ == '__main__':
    llm = OpenAI()
    resp = llm.complete("2+3?")
    print(resp)