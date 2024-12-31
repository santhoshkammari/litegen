from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional
from langchain_ollama import ChatOllama
import numpy as np


@dataclass
class SearchCandidate:
    query: str
    result: Optional[str] = None
    score: float = 0.0
    validation: str = ""


class TTCSearchValidator:
    def __init__(self, llm=None, n_candidates=4, beam_width=2):
        self.llm = llm or ChatOllama(model="llama2:latest")
        self.n_candidates = n_candidates  # Number of initial candidates
        self.beam_width = beam_width  # Number of best candidates to keep

    def search_google(self, query: str):
        """Mock Google search - replace with your actual search function"""
        return f"Sample result for: {query}"

    def generate_query_candidates(self, content: str) -> List[SearchCandidate]:
        """Generate multiple query candidates using temperature sampling"""
        prompt = f"""Generate a Google search query to validate this statement: '{content}'
                    Make the query specific and focused on key facts."""

        candidates = []
        for _ in range(self.n_candidates):
            response = self.llm.invoke(prompt, temperature=0.8)
            candidates.append(SearchCandidate(query=response.content))
        return candidates

    def score_candidate(self, candidate: SearchCandidate) -> float:
        """Score a candidate using the reward model (LLM)"""
        prompt = f"""Rate this Google search query and result pair for fact validation:
                    Query: {candidate.query}
                    Result: {candidate.result}

                    Rate from 0 to 1 where:
                    0.0 = Irrelevant or misleading
                    0.3 = Somewhat related but not specific
                    0.5 = Related but incomplete
                    0.8 = Good match with specific information
                    1.0 = Perfect match with authoritative information

                    Provide only the numerical score."""

        response = self.llm.invoke(prompt, temperature=0.2)
        try:
            return float(response.content.strip())
        except:
            return 0.0

    def validate_with_beam_search(self, content: str) -> SearchCandidate:
        """Use beam search to find the best query"""
        # Generate initial candidates
        candidates = self.generate_query_candidates(content)

        # Perform search for each candidate
        with ThreadPoolExecutor(max_workers=4) as executor:
            for candidate in candidates:
                candidate.result = self.search_google(candidate.query)

        # Score initial candidates
        for candidate in candidates:
            candidate.score = self.score_candidate(candidate)

        # Keep top beam_width candidates
        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)[:self.beam_width]

        # Refine top candidates
        for candidate in candidates:
            refine_prompt = f"""Improve this search query based on the results:
                              Original query: {candidate.query}
                              Results: {candidate.result}
                              Score: {candidate.score}

                              Provide only the improved query."""

            improved_query = self.llm.invoke(refine_prompt, temperature=0.5).content
            new_candidate = SearchCandidate(query=improved_query)
            new_candidate.result = self.search_google(new_candidate.query)
            new_candidate.score = self.score_candidate(new_candidate)

            if new_candidate.score > candidate.score:
                candidate.query = new_candidate.query
                candidate.result = new_candidate.result
                candidate.score = new_candidate.score

        # Return the best candidate
        return max(candidates, key=lambda x: x.score)

    def validate_content(self, content: str) -> dict:
        """Main validation method using TTC"""
        best_candidate = self.validate_with_beam_search(content)

        # Final validation using the best candidate
        validation_prompt = f"""Based on the search results, validate this statement:
                              Statement: {content}
                              Best search query: {best_candidate.query}
                              Search results: {best_candidate.result}

                              Provide a concise validation summary."""

        validation = self.llm.invoke(validation_prompt).content

        return {
            "original_content": content,
            "best_query": best_candidate.query,
            "search_result": best_candidate.result,
            "confidence_score": best_candidate.score,
            "validation": validation
        }


# Example usage
if __name__ == "__main__":
    from litegen.model import qwen2p5_7b
    validator = TTCSearchValidator(n_candidates=4, beam_width=2,llm=qwen2p5_7b(True))
    result = validator.validate_content("Bananas are not classified as berries, while strawberries are true berries.")
    print("Validation Result:")
    for key, value in result.items():
        print(f"{key}: {value}")