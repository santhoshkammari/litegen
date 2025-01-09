import os

from litegen.utils import get_func_from_response
from litegen import genai, completion

os.environ['OPENAI_BASE_URL'] = 'http://192.168.170.76:11434/v1'
os.environ['OPENAI_MODEL_NAME'] = "llama3.1:8b"


def get_weather(city: str) -> str:
    """Get the current weather for a city"""
    return f"its very cool 12 degrer in {city}"

def get_joke() -> str:
    """Get a random joke"""
    return genai(model="llama3.2:3b-instruct-q4_K_M", prompt="Tell me a joke")


# res = completion(
#     messages="tell me joke",
#     tools=[get_weather,get_joke]
# )

res = genai("tell me a joke")
print(res)

# print(get_func_from_response(res))
