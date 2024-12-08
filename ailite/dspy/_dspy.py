import os

# from dotenv import load_dotenv
# load_dotenv("../../.env")
# HF_API_KEY = os.environ.get("HF_API_KEY")

from typing import Optional, List, Any, Dict, Literal
from datetime import datetime
import uuid
import litellm
import httpx
from dspy import BaseLM
from dspy.utils.callback import BaseCallback

from ailitellm import HFModelType

class HFLM(BaseLM):
    """Custom LM class for Hugging Face models using LiteLLM"""

    def __init__(
        self,
        model: HFModelType = "Qwen/Qwen2.5-72B-Instruct",
        api_key: str = "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",
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        callbacks: Optional[List[BaseCallback]] = None,
        num_retries: int = 3,
        **kwargs,
    ):
        # Format the model name for Hugging Face
        if not model.startswith("huggingface/"):
            model = f"huggingface/{model}"

        self.model = model
        self.api_key = api_key
        self.model_type = model_type
        self.cache = cache
        self.callbacks = callbacks or []
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []
        self.num_retries = num_retries

    def _prepare_messages(self, prompt=None, messages=None):
        """Prepare messages for the API call"""
        if messages is None and prompt is not None:
            return [{"role": "user", "content": prompt}]
        return messages or []

    def _process_response(self, response: Any) -> List[str]:
        """Process the API response and extract outputs"""
        try:
            if hasattr(response, 'choices'):
                outputs = []
                for choice in response.choices:
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        outputs.append(choice.message.content)
                    elif hasattr(choice, 'text'):
                        outputs.append(choice.text)
                return outputs
            return [str(response)]
        except Exception as e:
            raise ValueError(f"Failed to process response: {str(e)}")

    def _log_interaction(self, prompt, messages, kwargs, outputs):
        """Log the interaction details"""
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": {k: v for k, v in kwargs.items() if not k.startswith("api_")},
            "outputs": outputs,
            "timestamp": datetime.now().isoformat(),
            "uuid": str(uuid.uuid4()),
            "model": self.model,
            "model_type": self.model_type,
        }
        self.history.append(entry)

    def __call__(self, prompt=None, messages=None, **kwargs):
        try:
            # Prepare messages and parameters
            messages = self._prepare_messages(prompt, messages)
            call_kwargs = {**self.kwargs, **kwargs}

            # Extract model name without the provider prefix for the API endpoint
            model_name = self.model.replace("huggingface/", "", 1)

            # Configure the API call
            completion_kwargs = {
                "model": self.model,
                "messages": messages,
                "api_key": self.api_key,
                "api_base": f"https://api-inference.huggingface.co/models/{model_name}",
                "temperature": call_kwargs.get('temperature', 0.0),
                "max_tokens": call_kwargs.get('max_tokens', 1000)
            }

            # Enable verbose logging for debugging
            litellm.set_verbose = False

            # Make the API call with retries
            for attempt in range(self.num_retries):
                try:
                    response = litellm.completion(**completion_kwargs)
                    outputs = self._process_response(response)
                    self._log_interaction(prompt, messages, call_kwargs, outputs)
                    return outputs
                except Exception as e:
                    if attempt == self.num_retries - 1:  # Last attempt
                        raise
                    continue

        except Exception as e:
            error_msg = f"Error during completion: {str(e)}"
            if isinstance(e, httpx.TimeoutException):
                error_msg = f"Timeout error: {str(e)}"
            elif isinstance(e, httpx.HTTPError):
                error_msg = f"HTTP error: {str(e)}"

            print(error_msg)
            raise Exception(error_msg) from e
