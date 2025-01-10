from litegen.agents import Agent

def test_agent_ollama():
    agent = Agent(model="smollm2:135m-instruct-q4_K_M",
          system_prompt="You are helpful assistant")
    response = agent("What is the capital of France?")
    assert response.content!=None

def agent_parallel_ollama():
    agent = Agent(model="smollm2:135m-instruct-q4_K_M",
          system_prompt="You are helpful assistant")
    response = agent.batch(["What is the capital of France?"]*5)

