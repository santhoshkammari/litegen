from litegen import genai
import os

os.environ['OPENAI_BASE_URL'] = 'http://192.168.170.76:11434/v1'

res = genai(model="qwen2.5:7b-instruct",
            messages="tell me joke in 5 lines")

print(res)