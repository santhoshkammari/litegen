from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import json
import asyncio
import uvicorn
import uuid

from..main._ailite import ai

def process_messages(messages):
    um = []
    for m in messages:
        um.append({"role": m.role, "content": m.content})
    return um



class ModelDetails(BaseModel):
    format: str = "gguf"
    family: str = "llama"
    families: Optional[List[str]] = None
    parameter_size: str = "7B"
    quantization_level: str = "Q4_0"


class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class GenerateRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    stream: bool = True
    raw: bool = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    context: Optional[List[int]] = None
    template: Optional[str] = None
    system: Optional[str] = None
    keep_alive: Optional[str] = "5m"


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"
    tools: Optional[List[Dict[str, Any]]] = None


class EmbedRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    options: Optional[Dict[str, Any]] = None


class OllamaAPI(FastAPI):
    def __init__(self):
        super().__init__(title="Ollama-like API")
        self.active_models = {}
        self.model_details = {}
        self.setup_routes()
        self.request_model_name = ""
        self.ai_models = [
    'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'CohereForAI/c4ai-command-r-plus-08-2024',
    'Qwen/Qwen2.5-72B-Instruct',
    'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    'Qwen/Qwen2.5-Coder-32B-Instruct',
    'meta-llama/Llama-3.2-11B-Vision-Instruct',
    'NousResearch/Hermes-3-Llama-3.1-8B',
    'mistralai/Mistral-Nemo-Instruct-2407',
    'microsoft/Phi-3.5-mini-instruct'
]

    def setup_routes(self):
        @self.post("/api/generate")
        async def generate(request: GenerateRequest):
            if request.stream:
                return StreamingResponse(
                    self._stream_generate(request),
                    media_type="text/event-stream"
                )
            return await self._generate(request)

        @self.post("/api/chat")
        async def chat(request: ChatRequest):
            if request.stream:
                return StreamingResponse(
                    self._stream_chat(request),
                    media_type="text/event-stream"
                )
            return await self._chat(request)

        # @self.post("/v1/chat/completions")
        # async def chat(request: ChatRequest):
        #     processed_messages = process_messages(request.messages)
        #     print(request.stream)
        #     print(request.messages)
        #     print('=======')
        #     if request.stream:
        #         return StreamingResponse(
        #             self._stream_chat(request,processed_messages=processed_messages),
        #             media_type="text/event-stream"
        #         )
        #     return await self._chat(request,processed_messages = processed_messages)

        @self.post("/api/embed")
        async def embed(request: EmbedRequest):
            return await self._embed(request)

        @self.get("/api/tags")
        async def list_models():
            return {"models": self._get_local_models()}

        @self.get("/api/ps")
        async def list_running():
            return {"models": self._get_running_models()}

    async def _generate(self, request: GenerateRequest) -> Dict[str, Any]:
        # Simulate model generation
        start_time = time.time()
        response = self.get_ai_response(request.prompt,request.model)
        self.request_model_name=request.model
        response = {
            "model": request.model,
            "created_at": datetime.utcnow().isoformat(),
            "response": response,
            "done": True,
            "context": [1, 2, 3],
            "total_duration": int((time.time() - start_time) * 1e9),
            "load_duration": 1000000,
            "prompt_eval_count": len(request.prompt) if request.prompt else 0,
            "prompt_eval_duration": 100000,
            "eval_count": 100,
            "eval_duration": 900000
        }
        return response

    async def _stream_generate(self, request: GenerateRequest):
        # Simulate streaming response
        for word in ai(model=request.model,prompt_or_messages=request.prompt,stream=True):
            await asyncio.sleep(0)
            response = {
                "model": request.model,
                "created_at": datetime.utcnow().isoformat(),
                "response": word,
                "done": False
            }
            yield f"data: {json.dumps(response)}\n\n"


    async def _chat(self, request: ChatRequest,processed_messages) -> Dict[str, Any]:
        # Simulate chat response
        start_time = time.time()
        ai_model_response = self.get_ai_response(processed_messages,request.model)
        response = {
            "model": request.model,
            "created_at": datetime.utcnow().isoformat(),
            "message": {
                "role": "assistant",
                "content": ai_model_response
            },
            "done": True,
            "total_duration": int((time.time() - start_time) * 1e9),
            "load_duration": 1000000,
            "prompt_eval_count": sum(len(m.content) for m in request.messages),
            "prompt_eval_duration": 100000,
            "eval_count": 100,
            "eval_duration": 900000
        }
        return response

    async def _stream_chat(self, request: ChatRequest,processed_messages):
        print('@@@@@@@')
        print(processed_messages)
        print('####3')
        for word in ai(prompt_or_messages=processed_messages,model=request.model,stream=True):
            print(word,end = "",flush=True)
            await asyncio.sleep(0)
            response = {
                "model": request.model,
                "created_at": str(uuid.uuid4()),
                "message": {"role": "assistant", "content": word},
                "done": False
            }
            yield f"data: {json.dumps(response)}\n\n"

    async def _embed(self, request: EmbedRequest) -> Dict[str, Any]:
        # Simulate embedding generation
        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = request.input

        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in inputs]

        return {
            "model": request.model,
            "embeddings": embeddings,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": sum(len(input_text) for input_text in inputs)
        }

    def _get_local_models(self) -> List[Dict[str, Any]]:
        # Simulate list of local models
        return [
            {
                "name": model,
                "modified_at": str(uuid4()),
                "size": 7365960935,
                "digest": "9f438cb9cd581fc025612d27f7c1a6669ff83a8bb0ed86c94fcf4c5440555697",
                "details": {
                    "format": "",
                    "family": "",
                    "families": None,
                    "parameter_size": "None",
                    "quantization_level": ""
                }
            } for model in self.ai_models
        ]

    def _get_running_models(self) -> List[Dict[str, Any]]:
        # Simulate list of running models
        return [
            {
                "name": self.request_model_name,
                "model": self.request_model_name,
                "size": 4000000000,
                "digest": str(uuid.uuid4()),
                "details": ModelDetails().dict(),
                "expires_at": (datetime.utcnow()).isoformat(),
                "size_vram": 4000000000
            }
        ]

    def get_ai_response(self, prompt,model):
        return ai(prompt_or_messages=prompt,model=model)



from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
import time
from datetime import datetime


# New models for OpenAI-style endpoints
class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None


# Update the OllamaAPI class with new endpoints
def add_openai_routes(app: OllamaAPI):
    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        # Convert OpenAI-style messages to Ollama format
        ollama_messages = [
            Message(
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls
            ) for msg in request.messages
        ]

        # Create Ollama-style chat request
        ollama_request = ChatRequest(
            model=request.model,
            messages=ollama_messages,
            stream=request.stream,
            options={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty
            },
            tools=request.tools
        )

        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(ollama_request),
                media_type="text/event-stream"
            )

        # Get response from existing chat implementation
        response = await app._chat(ollama_request)

        # Convert to OpenAI format
        return ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response["message"]["content"]
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": response["prompt_eval_count"],
                "completion_tokens": response["eval_count"],
                "total_tokens": response["prompt_eval_count"] + response["eval_count"]
            }
        )

    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        # Convert to Ollama generate request
        ollama_request = GenerateRequest(
            model=request.model,
            prompt=request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt),
            stream=request.stream,
            options={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty
            }
        )

        if request.stream:
            return StreamingResponse(
                _stream_completion(ollama_request),
                media_type="text/event-stream"
            )

        # Get response from existing generate implementation
        response = await app._generate(ollama_request)

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": response["response"],
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": response["prompt_eval_count"],
                "completion_tokens": response["eval_count"],
                "total_tokens": response["prompt_eval_count"] + response["eval_count"]
            }
        }

    async def _stream_chat_completion(request: ChatRequest):
        async for chunk in app._stream_chat(request,process_messages(request.messages)):
            data = json.loads(chunk.replace("data: ", ""))
            if data["done"]:
                continue

            yield f"""data: {json.dumps({
                'id': f"chatcmpl-{int(time.time())}",
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': request.model,
                'choices': [{
                    'index': 0,
                    'delta': {
                        'role': 'assistant',
                        'content': data['message']['content']
                    },
                    'finish_reason': None
                }]
            })}\n\n"""

    async def _stream_completion(request: GenerateRequest):
        async for chunk in app._stream_generate(request):
            data = json.loads(chunk.replace("data: ", ""))
            if data["done"]:
                continue

            yield f"""data: {json.dumps({
                'id': f"cmpl-{int(time.time())}",
                'object': 'text_completion',
                'created': int(time.time()),
                'model': request.model,
                'choices': [{
                    'text': data['response'],
                    'index': 0,
                    'logprobs': None,
                    'finish_reason': None
                }]
            })}\n\n"""

    # Add models endpoint
    @app.get("/v1/models")
    async def list_models():
        models = app._get_local_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": model["name"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "organization",
                    "permission": [],
                    "root": model["name"],
                    "parent": None
                }
                for model in models
            ]
        }


# Update the OllamaAPI class initialization
def initialize_openai_routes(app: OllamaAPI):
    add_openai_routes(app)

def ailite_ollama_api(host="0.0.0.0", port=11436):
    app = OllamaAPI()
    initialize_openai_routes(app)  # Add this line to initialize OpenAI routes
    uvicorn.run(app, host=host,port=port)

# Update the main block
if __name__ == "__main__":
    ailite_ollama_api()
