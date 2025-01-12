from litegen import genai
import os

os.environ['OPENAI_API_KEY'] = 'huggingchat'
os.environ['OPENAI_MODEL_NAME'] = "NousResearch/Hermes-3-Llama-3.1-8B"

queries = [f'{a}+{b}?, return only integer' for a,b in zip(range(1,10),range(1,10))]

res = [genai(x) for x in queries]
print(res)
