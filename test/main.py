# from litegen import pp_completion
#
# pp_completion(model='smollm2:135m-instruct-q4_K_M',
#                  messages="tell me about narendra modi in 5 paragrapsh simple")
#

from litegen import genai
import os

os.environ['OPENAI_API_KEY'] = 'huggingchat'

res = genai(model="NousResearch/Hermes-3-Llama-3.1-8B",
            messages="tell me about narendra modi in 5 paragrapsh simple")

print(res)