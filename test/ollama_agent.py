from litegen import Agent
from litegen.trace_llm import TraceLLM
llm_tracer = TraceLLM()
llm_tracer.set_experiment("dummy")

agent = Agent(model='smollm2:135m-instruct-q4_K_M',
              llm_tracer=llm_tracer)

print(agent('hai'))
