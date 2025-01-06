from litegen import genai
import os

# os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_BASE_URL']  = "huggingchat"

res = genai(model="qwen2.5:7b-instruct",
            messages="tell me joke in 5 lines")

print(res)