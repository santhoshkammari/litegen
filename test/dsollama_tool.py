from litegen import genai, completion
import os

os.environ['OPENAI_BASE_URL'] = 'http://192.168.170.76:11434/v1'

def current_weather(city:str) -> str:
    """Get the current weather for a city"""
    return f"its very cool 12 degrer in {city}"

tools=[{
  'type': 'function',
  'function': {
    'name': 'current_weather',
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
res = completion(model="llama3.2:3b-instruct-fp16",
            messages="tell me weather in hyderabad",
            tools=[current_weather]
                 # tools=tools
                 )

print(res.choices[0].message.tool_calls)

