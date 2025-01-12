from litegen import genai
import os

os.environ['OPENAI_API_KEY'] = 'huggingchat'

res = genai(model="NousResearch/Hermes-3-Llama-3.1-8B",
            messages="what is your name?",
            system_prompt='you are doctor and your name is kammari santhosh, you work in yashnu hospitals')
print(res)
# for r in res:
#     print(r)