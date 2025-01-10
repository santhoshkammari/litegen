import asyncio

from litegen.agents import Agent

agent = Agent(model="smollm2:135m-instruct-q4_K_M",
              system_prompt="You are helpful assistant")
response = agent('what is capital of france')

async def agent_parallel_ollama(n:int):

    response = await agent.batch(["What is the capital of japan?"]*n,max_tokens=1)
    for i in response:
        print('-------')
        print(i.content)

if __name__ == '__main__':

    asyncio.run(agent_parallel_ollama())