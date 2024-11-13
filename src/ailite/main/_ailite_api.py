import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from fastapi.responses import StreamingResponse

from ._model._api._client import HUGPIClient
from ..prompts import AILITE_X_CLAUDE_PROMPT
from ._model._api.types._model_types import MODELS_TYPE

app = FastAPI()

client = None

# Initialize the HuggyLLM instance
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    conversation: Optional[bool] = False
    stream: Optional[bool] = False
    websearch:Optional[bool] = False

class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    conversation: Optional[bool] = False
    stream: Optional[bool] = False
    websearch: Optional[bool] = False
    assistant: Optional[str] = None

@app.post("/v1/chat")
async def chat(request: ChatRequest):
    # try:
    messages = [msg.dict() for msg in request.messages]
    if request.stream:
        return StreamingResponse(
            stream_chatbot_response(request),
            media_type="text/event-stream"
        )
    else:
        response = client.messages.create(messages = messages, model_name=request.model, conversation=request.conversation,
                                          web_search = request.websearch)
        return {"message": {"content": response.content[0]["text"]}}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    # try:
    if request.stream:
        return StreamingResponse(
            stream_chatbot_response(request),
            media_type="text/event-stream"
        )
    else:
        response = client.messages.create(prompt = request.prompt, model_name=request.model, conversation=request.conversation,
                                          web_search = request.websearch)
        return {"message": {"content": response.content[0]["text"]}}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

async def stream_chatbot_response(request:GenerateRequest):
    for chunk in client.messages.create(messages=request.prompt, model_name=request.model, conversation=request.conversation,
                                        web_search=request.websearch,
                                        stream=True):
        yield chunk.content[0]['text']
        await asyncio.sleep(0)  # Allow other tasks to run

def serve(
    model:MODELS_TYPE = 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    system_prompt:Optional[str]=None
):
    import uvicorn
    global client
    if client is None:
        client = HUGPIClient(model=model,system_prompt=system_prompt or AILITE_X_CLAUDE_PROMPT)
    uvicorn.run(app, host="0.0.0.0", port=11435)

if __name__ == "__main__":
    serve()
