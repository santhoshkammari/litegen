import os
from litegen import Agent

# from litegen.trace_llm import TraceLLM
# llm_tracer = TraceLLM()
# llm_tracer.set_experiment('Qwen2.6')


os.environ['OPENAI_API_KEY'] = 'huggingchat'
os.environ['OPENAI_TRACING'] = 'true'

agent = Agent(model='microsoft/Phi-3.5-mini-instruct')

res = agent("tell me about modi in 100 words")

print(res.content)
