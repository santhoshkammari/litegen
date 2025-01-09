import inspect
import re
from collections import defaultdict
from typing import Callable, Optional, Any, Literal, Sequence, Mapping, Union

from pydantic import BaseModel, ConfigDict

class SubscriptableBaseModel(BaseModel):
  def __getitem__(self, key: str) -> Any:
    """
    >>> msg = Message(role='user')
    >>> msg['role']
    'user'
    >>> msg = Message(role='user')
    >>> msg['nonexistent']
    Traceback (most recent call last):
    KeyError: 'nonexistent'
    """
    if key in self:
      return getattr(self, key)

    raise KeyError(key)

  def __setitem__(self, key: str, value: Any) -> None:
    """
    >>> msg = Message(role='user')
    >>> msg['role'] = 'assistant'
    >>> msg['role']
    'assistant'
    >>> tool_call = Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))
    >>> msg = Message(role='user', content='hello')
    >>> msg['tool_calls'] = [tool_call]
    >>> msg['tool_calls'][0]['function']['name']
    'foo'
    """
    setattr(self, key, value)

  def __contains__(self, key: str) -> bool:
    """
    >>> msg = Message(role='user')
    >>> 'nonexistent' in msg
    False
    >>> 'role' in msg
    True
    >>> 'content' in msg
    False
    >>> msg.content = 'hello!'
    >>> 'content' in msg
    True
    >>> msg = Message(role='user', content='hello!')
    >>> 'content' in msg
    True
    >>> 'tool_calls' in msg
    False
    >>> msg['tool_calls'] = []
    >>> 'tool_calls' in msg
    True
    >>> msg['tool_calls'] = [Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))]
    >>> 'tool_calls' in msg
    True
    >>> msg['tool_calls'] = None
    >>> 'tool_calls' in msg
    True
    >>> tool = Tool()
    >>> 'type' in tool
    True
    """
    if key in self.model_fields_set:
      return True

    if key in self.model_fields:
      return self.model_fields[key].default is not None

    return False

  def get(self, key: str, default: Any = None) -> Any:
    """
    >>> msg = Message(role='user')
    >>> msg.get('role')
    'user'
    >>> msg = Message(role='user')
    >>> msg.get('nonexistent')
    >>> msg = Message(role='user')
    >>> msg.get('nonexistent', 'default')
    'default'
    >>> msg = Message(role='user', tool_calls=[ Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))])
    >>> msg.get('tool_calls')[0]['function']['name']
    'foo'
    """
    return self[key] if key in self else default

class Tool(SubscriptableBaseModel):
  type: Optional[Literal['function']] = 'function'

  class Function(SubscriptableBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

    class Parameters(SubscriptableBaseModel):
      type: Optional[Literal['object']] = 'object'
      required: Optional[Sequence[str]] = None

      class Property(SubscriptableBaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        type: Optional[str] = None
        description: Optional[str] = None

      properties: Optional[Mapping[str, Property]] = None

    parameters: Optional[Parameters] = None

  function: Optional[Function] = None


def _parse_docstring(doc_string: Union[str, None]) -> dict[str, str]:
  parsed_docstring = defaultdict(str)
  if not doc_string:
    return parsed_docstring

  key = hash(doc_string)
  for line in doc_string.splitlines():
    lowered_line = line.lower().strip()
    if lowered_line.startswith('args:'):
      key = 'args'
    elif lowered_line.startswith('returns:') or lowered_line.startswith('yields:') or lowered_line.startswith('raises:'):
      key = '_'

    else:
      # maybe change to a list and join later
      parsed_docstring[key] += f'{line.strip()}\n'

  last_key = None
  for line in parsed_docstring['args'].splitlines():
    line = line.strip()
    if ':' in line:
      # Split the line on either:
      # 1. A parenthetical expression like (integer) - captured in group 1
      # 2. A colon :
      # Followed by optional whitespace. Only split on first occurrence.
      parts = re.split(r'(?:\(([^)]*)\)|:)\s*', line, maxsplit=1)

      arg_name = parts[0].strip()
      last_key = arg_name

      # Get the description - will be in parts[1] if parenthetical or parts[-1] if after colon
      arg_description = parts[-1].strip()
      if len(parts) > 2 and parts[1]:  # Has parenthetical content
        arg_description = parts[-1].split(':', 1)[-1].strip()

      parsed_docstring[last_key] = arg_description

    elif last_key and line:
      parsed_docstring[last_key] += ' ' + line

  return parsed_docstring

def convert_function_to_tool(func: Callable) -> Tool:
  doc_string_hash = hash(inspect.getdoc(func))
  parsed_docstring = _parse_docstring(inspect.getdoc(func))
  schema = type(
    func.__name__,
    (BaseModel,),
    {
      '__annotations__': {k: v.annotation if v.annotation != inspect._empty else str for k, v in inspect.signature(func).parameters.items()},
      '__signature__': inspect.signature(func),
      '__doc__': parsed_docstring[doc_string_hash],
    },
  ).model_json_schema()

  for k, v in schema.get('properties', {}).items():
    # If type is missing, the default is string
    types = {t.get('type', 'string') for t in v.get('anyOf')} if 'anyOf' in v else {v.get('type', 'string')}
    if 'null' in types:
      schema['required'].remove(k)
      types.discard('null')

    schema['properties'][k] = {
      'description': parsed_docstring[k],
      'type': ', '.join(types),
    }

  tool = Tool(
    function=Tool.Function(
      name=func.__name__,
      description=schema.get('description', ''),
      parameters=Tool.Function.Parameters(**schema),
    )
  )

  return Tool.model_validate(tool)