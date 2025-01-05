from litegen import genai
import os

os.environ['OPENAI_API_KEY'] = 'huggingchat'

res = genai(model="NousResearch/Hermes-3-Llama-3.1-8B",
            messages="tell me about narendra modi in 5 paragrapsh simple")

print(res)