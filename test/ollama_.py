from litegen import genai
import os

os.environ['OPENAI_API_KEY'] = 'ollama'

res = genai(model="smollm2:135m-instruct-q4_K_M",
            messages="what is 2+3?")

print(res)