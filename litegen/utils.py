from typing import List, Dict, Any, Optional, Union
from openai.types.chat import ChatCompletionMessageParam
import json


def create_system_message(content: str) -> Dict[str, str]:
    """Create a system message."""
    return {"role": "system", "content": content}


def create_user_message(content: str) -> Dict[str, str]:
    """Create a user message."""
    return {"role": "user", "content": content}


def create_assistant_message(content: str) -> Dict[str, str]:
    """Create an assistant message."""
    return {"role": "assistant", "content": content}


def create_tool_message(content: str, tool_call_id: str) -> Dict[str, str]:
    """Create a tool message."""
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format chat history into a readable string."""
    formatted = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def extract_json_from_string(text: str) -> Dict:
    """Extract JSON from a string that might contain other text."""
    try:
        # Try to find JSON-like content between curly braces
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return {}


def create_function_json(name: str, description: str, parameters: Dict) -> Dict:
    """Create a function definition in OpenAI's format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys())
            }
        }
    }


def format_code_response(code: str, language: str = "python") -> str:
    """Format code with markdown code block."""
    return f"```{language}\n{code}\n```"


def chunk_string(text: str, max_length: int = 2000) -> List[str]:
    """Split a string into chunks of maximum length."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def clean_string(text: str) -> str:
    """Clean a string for API use (remove extra whitespace, normalize quotes)."""
    return " ".join(text.replace('"', '"').replace('"', '"').split())


def format_list_to_prompt(items: List[str], prefix: str = "- ") -> str:
    """Format a list into a prompt-friendly string."""
    return "\n".join(f"{prefix}{item}" for item in items)


def format_dict_to_prompt(data: Dict) -> str:
    """Format a dictionary into a prompt-friendly string."""
    return "\n".join(f"{key}: {value}" for key, value in data.items())


def create_few_shot_prompt(examples: List[Dict[str, str]],
                           input_key: str = "input",
                           output_key: str = "output") -> str:
    """Create a few-shot prompt from examples."""
    formatted = []
    for ex in examples:
        formatted.extend([
            f"Input: {ex[input_key]}",
            f"Output: {ex[output_key]}",
            ""
        ])
    return "\n".join(formatted)


def format_error_response(error: str) -> Dict[str, str]:
    """Format an error message in a consistent way."""
    return {
        "error": True,
        "message": error
    }


def combine_messages(*message_lists: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Combine multiple message lists while preserving order."""
    combined = []
    for messages in message_lists:
        combined.extend(messages)
    return combined


def create_structured_prompt(context: str,
                             instruction: str,
                             examples: List[str] = None) -> str:
    """Create a structured prompt with context, instruction, and optional examples."""
    parts = [
        f"Context: {context}",
        f"Instruction: {instruction}"
    ]

    if examples:
        examples_str = "\n".join(f"- {ex}" for ex in examples)
        parts.append(f"Examples:\n{examples_str}")

    return "\n\n".join(parts)


def parse_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
    """Parse tool calls into a simpler format."""
    parsed = []
    for call in tool_calls:
        parsed.append({
            "name": call["function"]["name"],
            "arguments": json.loads(call["function"]["arguments"]),
            "id": call["id"]
        })
    return parsed


def format_json_schema(properties: Dict[str, Dict]) -> Dict:
    """Create a JSON schema for structured outputs."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys())
    }


def create_chat_prompt(system: str, user: str) -> List[Dict[str, str]]:
    """Create a basic chat prompt with system and user messages."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

def get_str_from_response(response):
    return response.choices[0].message.content

def get_func_from_response(response):
    return response.choices[0].message.tool_calls[0].function