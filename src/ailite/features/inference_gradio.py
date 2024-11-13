from typing import List, Tuple, Optional, Literal
from gradio_client import Client
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Represents a chat message with its response"""
    query: str
    response: str


class GradioInference:
    """
    A user-friendly wrapper for interacting with the Qwen2.5-Coder Gradio API.
    """

    MODEL_SIZES = Literal['0.5B', '1.5B', '3B', '7B', '14B', '32B']
    DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    def __init__(self,
                 model_name_or_url:str = "Qwen/Qwen2.5-Coder-demo",
                 model_size: MODEL_SIZES = "32B",
                 system_prompt: Optional[str] = None):
        """
        Initialize the Qwen model interface.

        Args:
            model_size: Size of the model to use (default: "32B")
            system_prompt: Custom system prompt (default: None, uses default prompt)
        """
        self.client = Client(model_name_or_url)
        self.history: List[Tuple[str, str]] = []
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.set_model_size(model_size)

    def set_model_size(self, model_size: MODEL_SIZES) -> None:
        """
        Change the model size.

        Args:
            model_size: New model size to use
        """
        result = self.client.predict(
            radio=model_size,
            system=self.system_prompt,
            api_name="/chiose_radio"
        )
        # Clear history when changing models
        self.clear_history()

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Update the system prompt.

        Args:
            system_prompt: New system prompt to use
        """
        self.system_prompt = system_prompt
        result = self.client.predict(
            system=system_prompt,
            api_name="/modify_system_session"
        )

    def chat(self, message: str) -> ChatMessage:
        """
        Send a message to the model and get its response.

        Args:
            message: The message to send to the model

        Returns:
            ChatMessage object containing the query and response
        """
        result = self.client.predict(
            query=message,
            history=self.history,
            system=self.system_prompt,
            api_name="/model_chat"
        )

        # Extract response from result tuple
        response = result[1][-1][1] if result[1] else ""

        # Update history
        self.history.append((message, response))

        return ChatMessage(query=message, response=response)

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.client.predict(api_name="/clear_session")
        self.history = []

    @property
    def chat_history(self) -> List[ChatMessage]:
        """
        Get the full chat history.

        Returns:
            List of ChatMessage objects
        """
        return [ChatMessage(query=q, response=r) for q, r in self.history]

if __name__ == '__main__':
    inference = GradioInference()
    res = inference.chat("How do i read a csv file in Python")
    print(res.response)