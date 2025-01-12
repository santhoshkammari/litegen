from litegen import Agent
import os

# os.environ['OPENAI_API_KEY'] = 'ollama'

agent = Agent(model='smollm2:135m-instruct-q4_K_M')

print(agent('hai'))
