import time

from litegen import Agent
from litegen.trace_llm import TraceLLM

llm_tracer = TraceLLM()
llm_tracer.set_experiment('Qwen2.5')

agent = Agent(model='qwen2.5:0.5b-instruct',
              llm_tracer=llm_tracer,
              name='DoctorQwen')

agent("what is 2+3?")
time.sleep(5)
agent("what is 4+5")
time.sleep(5)
agent("what is 6+7")
time.sleep(5)

