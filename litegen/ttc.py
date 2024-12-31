import openai
from typing import List, Dict, Optional, Union, Tuple
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for test-time compute scaling"""
    temperature: float = 0.8
    max_tokens: int = 2048
    num_generations: int = 256
    beam_width: int = 4
    max_steps: int = 40
    model: str = "gpt-4-turbo-preview"  # Model for generations
    reward_model: str = "gpt-4"  # Model for scoring


class TestTimeScaling:
    def __init__(self, api_key: str, config: Config):
        """Initialize the test-time compute scaling framework"""
        openai.api_key = api_key
        self.config = config

    async def generate_completion(self, prompt: str) -> str:
        """Generate a single completion using OpenAI API"""
        try:
            response = await openai.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return ""

    async def generate_batch(self, prompt: str, n: int) -> List[str]:
        """Generate multiple completions in parallel"""
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self.generate_completion(prompt))
                for _ in range(n)
            ]
        return [t.result() for t in tasks]

    async def score_solution(self, problem: str, solution: str) -> float:
        """Score a solution using the reward model"""
        prompt = f"""Problem: {problem}

Solution: {solution}

On a scale from 0 to 1, how likely is this solution to be correct? 
Respond with only a number between 0 and 1."""

        try:
            response = await openai.chat.completions.create(
                model=self.config.reward_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score_str = response.choices[0].message.content.strip()
            return float(score_str)
        except Exception as e:
            logger.error(f"Error scoring solution: {e}")
            return 0.0

    async def majority_voting(self, problem: str) -> str:
        """Implementation of majority voting strategy"""
        logger.info("Running majority voting strategy...")

        # Generate multiple solutions
        solutions = await self.generate_batch(problem, self.config.num_generations)

        # Count occurrences of each unique solution
        solution_counts = defaultdict(int)
        for solution in solutions:
            solution_counts[solution.strip()] += 1

        # Return the most common solution
        if not solution_counts:
            return ""

        return max(solution_counts.items(), key=lambda x: x[1])[0]

    async def best_of_n(self, problem: str, weighted: bool = False) -> str:
        """Implementation of Best-of-N strategy"""
        logger.info(f"Running Best-of-N strategy (weighted={weighted})...")

        # Generate solutions
        solutions = await self.generate_batch(problem, self.config.num_generations)

        # Score solutions
        scores = []
        for solution in solutions:
            score = await self.score_solution(problem, solution)
            scores.append((solution, score))

        if not scores:
            return ""

        if weighted:
            # Group identical solutions and sum their scores
            solution_scores = defaultdict(float)
            for solution, score in scores:
                solution_scores[solution.strip()] += score
            return max(solution_scores.items(), key=lambda x: x[1])[0]
        else:
            # Return solution with highest individual score
            return max(scores, key=lambda x: x[1])[0]

    async def beam_search(self, problem: str) -> str:
        """Implementation of beam search with process reward model"""
        logger.info("Running beam search strategy...")

        beams = [(problem, [], 1.0)]  # (context, steps, cumulative_score)

        for step in range(self.config.max_steps):
            new_beams = []

            for context, steps, cum_score in beams:
                # Generate next steps
                next_steps = await self.generate_batch(
                    context,
                    self.config.beam_width
                )

                for next_step in next_steps:
                    new_step_list = steps + [next_step]
                    score = await self.score_solution(
                        problem,
                        "\n".join(new_step_list)
                    )
                    new_cum_score = cum_score * score
                    new_beams.append((
                        context + "\n" + next_step,
                        new_step_list,
                        new_cum_score
                    ))

            # Select top beams
            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)
            beams = beams[:self.config.num_generations]

            # Check if we have a complete solution
            if any("Therefore, the answer is" in b[0] for b in beams):
                break

        # Return best complete solution
        return beams[0][0] if beams else ""

    async def dvts(self, problem: str) -> str:
        """Implementation of Diverse Verifier Tree Search"""
        logger.info("Running DVTS strategy...")

        num_subtrees = self.config.num_generations // self.config.beam_width
        subtrees = []

        # Initialize subtrees with diverse starting points
        initial_solutions = await self.generate_batch(problem, num_subtrees)

        for solution in initial_solutions:
            score = await self.score_solution(problem, solution)
            subtrees.append((problem, [solution], score))

        # Expand each subtree independently
        final_solutions = []

        for context, steps, _ in subtrees:
            # Run mini beam search for each subtree
            beam = await self.beam_search_subtree(
                context,
                steps,
                self.config.beam_width
            )
            if beam:
                final_solutions.append(beam)

        if not final_solutions:
            return ""

        # Return best solution across all subtrees
        return max(final_solutions, key=lambda x: x[2])[0]

    async def beam_search_subtree(
        self,
        context: str,
        steps: List[str],
        beam_width: int
    ) -> Optional[Tuple[str, List[str], float]]:
        """Helper function for DVTS to expand a single subtree"""
        beam = [(context, steps, 1.0)]

        for _ in range(self.config.max_steps):
            new_beam = []

            for ctx, step_list, cum_score in beam:
                candidates = await self.generate_batch(ctx, beam_width)

                for candidate in candidates:
                    new_steps = step_list + [candidate]
                    score = await self.score_solution(
                        context,
                        "\n".join(new_steps)
                    )
                    new_cum_score = cum_score * score
                    new_beam.append((
                        ctx + "\n" + candidate,
                        new_steps,
                        new_cum_score
                    ))

            # Select best beam
            beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:1]

            # Check if we have a complete solution
            if "Therefore, the answer is" in beam[0][0]:
                return beam[0]

        return None

    async def solve(self,
                    problem: str,
                    strategy: str = "beam_search") -> str:
        """Main interface to solve problems using different strategies"""
        strategies = {
            "majority": self.majority_voting,
            "best_of_n": self.best_of_n,
            "weighted_best_of_n": lambda p: self.best_of_n(p, weighted=True),
            "beam_search": self.beam_search,
            "dvts": self.dvts
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        return await strategies[strategy](problem)


# Example usage
async def main():
    config = Config(
        temperature=0.8,
        max_tokens=2048,
        num_generations=256,
        beam_width=4,
        max_steps=40
    )

    scaling = TestTimeScaling("your-api-key", config)

    problem = """Question: A train travels at an average speed of 60 kilometers per hour. 
    If it covers a distance of 300 kilometers, how many hours does it take?"""

    # Try each strategy
    strategies = [
        "majority",
        "best_of_n",
        "weighted_best_of_n",
        "beam_search",
        "dvts"
    ]

    for strategy in strategies:
        print(f"\nTrying {strategy}...")
        solution = await scaling.solve(problem, strategy)
        print(f"Solution: {solution}")


if __name__ == "__main__":
    asyncio.run(main())