import os

import openai

os.environ['OPENAI_BASE_URL'] = 'http://192.168.170.76:11434/v1'
openai.api_key = 'ollama'

tools=[{
  'type': 'function',
  'function': {
    'name': 'get_current_weather',
    'description': 'Get the current weather for a city',
    'parameters': {
      'type': 'object',
      'properties': {
        'city': {
          'type': 'string',
          'description': 'The name of the city',
        },
      },
      'required': ['city'],
    },
  },
},
]

messages=[{'role': 'user', 'content':
        'What is the weather in Toronto?'}]


response = openai.chat.completions.create(
	model="llama3.2:3b-instruct-fp16",
	messages=messages,
	tools=tools
)

print(response)