import dspy
import litellm
from ailitellm import HFModelType

class HFLM(dspy.LM):
    """HuggingFace Language Model integration for DSPy using LiteLLM"""

    def __init__(
        self,
        model: HFModelType,
        api_key: str = "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",
        model_type: str = "chat",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        cache: bool = True,
        **kwargs
    ):
        # Configure litellm
        litellm.api_key = api_key
        litellm.api_base = "https://api-inference.huggingface.co/v1/"

        # Initialize parent LM class with huggingface/ prefix
        super().__init__(
            model=f"huggingface/{model}",
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            api_key=api_key,
            **kwargs
        )

        # Store HF-specific settings
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co/v1/"