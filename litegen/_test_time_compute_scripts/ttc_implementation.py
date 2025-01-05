import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass
class Config:
    """Configuration for test-time compute scaling"""
    temperature: float = 0.8
    num_generations: int = 4
    model: str = 'qwen2.5:7b-instruct'
    system_prompt:str=""


class TestTimeComputeScaling:
    def __init__(self, config: Config, genai: Callable):
        """Initialize the test-time compute scaling framework"""
        self.config = config
        self.genai = genai  # Function to generate completions using OpenAI API
        self.strats = {
            "majority_voting": self.majority_voting,
            # "best_of_n": self.best_of_n,
            # "beam_search": self.beam_search,
        }

    async def _gen_query(self, query):
        """Async wrapper for generation function"""
        return self.genai(
            model=self.config.model,
            prompt=query,
            temperature=self.config.temperature,
            system_prompt=self.config.system_prompt
        )

    async def majority_voting(self, query):
        # Implementation of majority voting strategy
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self._gen_query(query))
                for _ in range(self.config.num_generations)
            ]

        results = [t.result() for t in tasks]

        solutions = defaultdict(int)
        for result in results:
            solutions[result.strip().lower()] += 1

        print(solutions)

        if not solutions:
            return ""
        return max(solutions.items(), key=lambda x: x[1])[0]

    async def run(self, query, strategy: Literal['majority_voting', 'best_of_n', 'beam_search']):
        return await self.strats[strategy](query)


if __name__ == '__main__':
    config = Config(
        temperature=0.8,
        num_generations=32,
        model='llama3.2:1b-instruct-fp16',
        system_prompt='You are mathematics expert, solve the problem and return answer directly'
    )
    from litegen._oai import OmniLLMClient

    genai = OmniLLMClient(gpu=True).completion

    scale = TestTimeComputeScaling(config=config, genai=genai)
    res = asyncio.run(scale.run('what is 12*25-32, return direct answer', strategy='majority_voting'))
    print(res)