# Litegen

Litegen is a lightweight Python wrapper for managing LLM interactions, supporting both local Ollama models and OpenAI API. It provides a simple, unified interface for chat completions with streaming capabilities.

## Installation

```bash
pip install litegen
```

## Features

- 🚀 Simple unified interface for LLM interactions
- 🤖 Support for both local Ollama models and OpenAI
- 📡 Built-in streaming capabilities
- 🛠 Function calling support
- 🔄 Context management for conversations
- 🎯 GPU support for enhanced performance

## Quick Start

```python
from litegen import completion, pp_completion

# Simple completion
response = completion(
    model="mistral",  # or any Ollama/OpenAI model
    prompt="What is the capital of France?"
)
print(response.choices[0].message.content)

# Streaming completion with pretty print
pp_completion(
    model="llama2",
    prompt="Write a short story about a robot",
    temperature=0.7
)
```

## Advanced Usage

### System Prompts and Context

```python
response = completion(
    model="mistral",
    system_prompt="You are a helpful math tutor",
    prompt="Explain the Pythagorean theorem",
    context=[
        {"role": "user", "content": "Can you help me with math?"},
        {"role": "assistant", "content": "Of course! What would you like to know?"}
    ]
)
```

### Function Calling

```python
def get_weather(location: str, unit: str = "celsius"):
    """Get weather for a location"""
    pass

response = completion(
    model="gpt-3.5-turbo",
    prompt="What's the weather in Paris?",
    tools=[get_weather]
)
```

### GPU Support

```python
response = completion(
    model="mistral",
    prompt="Complex calculation task",
    gpu=True  # Enable GPU acceleration
)
```

## Configuration

The client can be configured with custom settings:

```python
from litegen import get_client

client = get_client(gpu=True)  # Enable GPU support
# Use client directly for more control
```

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional, for OpenAI models)

## API Reference

### completion(...)

Main function for chat completions.

```python
completion(
    model: str,                    # Model name
    messages: Optional[List[Dict[str, str]]] | str = None,  # Raw messages or prompt string
    system_prompt: str = "You are helpful Assistant",  # System prompt
    prompt: str = "",              # User prompt
    context: Optional[List[Dict[str, str]]] = None,  # Conversation history
    temperature: Optional[float] = None,  # Temperature for response randomness
    max_tokens: Optional[int] = None,  # Max tokens in response
    stream: bool = False,          # Enable streaming
    stop: Optional[List[str]] = None,  # Stop sequences
    tools: Optional[List] = None,  # Function calling tools
    gpu: bool = False,            # Enable GPU
    **kwargs                      # Additional parameters
)
```

### pp_completion(...)

Streaming-enabled completion with pretty printing. Takes the same parameters as `completion()`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License