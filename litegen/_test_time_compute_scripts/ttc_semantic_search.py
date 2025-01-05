from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional
from wordllama import WordLlama
from searchlite import google
from parselite import parse
import numpy as np


@dataclass
class SearchResult:
    query: str
    urls: List[str]
    contents: List[str]
    score: float = 0.0
    embedding: Optional[np.ndarray] = None


class TTCSemanticSearch:
    def __init__(self, n_candidates=4, beam_width=2):
        # Initialize WordLlama for semantic understanding
        self.wllm = WordLlama.load()
        self.n_candidates = n_candidates  # Number of initial query variants
        self.beam_width = beam_width  # Number of best candidates to keep

    def generate_query_variants(self, base_query: str) -> List[str]:
        """Generate query variations using WordLlama semantic understanding"""
        # Create slightly different versions of the query
        chunks = self.wllm.split(base_query, target_size=256)
        variants = [base_query]  # Start with original query

        # Generate variations by focusing on different semantic chunks
        for chunk in chunks:
            similar_docs = self.wllm.rank(chunk, [base_query], batch_size=1)
            if similar_docs:
                variants.append(similar_docs[0][0])

        # Ensure we have enough variants
        while len(variants) < self.n_candidates:
            variants.append(base_query)

        return variants[:self.n_candidates]

    def search_and_parse(self, query: str) -> SearchResult:
        """Perform search and parse results for a single query"""
        try:
            urls = google(query)[:5]  # Get top 5 results
            results = parse(urls)
            contents = [r.content for r in results if r.content]

            # Get embedding for the combined content
            combined_content = " ".join(contents[:3])  # Use top 3 results for embedding
            embedding = self.wllm.embed([combined_content])[0] if combined_content else None

            return SearchResult(
                query=query,
                urls=urls,
                contents=contents,
                embedding=embedding
            )
        except Exception as e:
            print(f"Error in search_and_parse for query '{query}': {e}")
            return SearchResult(query=query, urls=[], contents=[])

    def score_result(self, result: SearchResult, original_query: str) -> float:
        """Score search results using WordLlama similarity"""
        if not result.contents or not result.embedding is not None:
            return 0.0

        # Get embedding for original query
        query_embedding = self.wllm.embed([original_query])[0]

        # Calculate similarity between query and results
        similarity = self.wllm.similarity(original_query, " ".join(result.contents[:3]))

        # Bonus score for more diverse results
        diversity_score = len(set(result.urls)) / max(len(result.urls), 1)

        return similarity * 0.8 + diversity_score * 0.2

    def beam_search_results(self, query: str) -> SearchResult:
        """Use beam search to find the best search results"""
        # Generate initial query candidates
        query_variants = self.generate_query_variants(query)

        # Search and parse results for each variant
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.search_and_parse, query_variants))

        # Score initial results
        for result in results:
            result.score = self.score_result(result, query)

        # Keep top beam_width results
        results = sorted(results, key=lambda x: x.score, reverse=True)[:self.beam_width]

        # Refine top results with additional query variations
        for result in results:
            if result.contents:
                # Generate new query based on top content
                refined_query = self.wllm.rank(query, result.contents[:1])[0][0]
                new_result = self.search_and_parse(refined_query)
                new_result.score = self.score_result(new_result, query)

                if new_result.score > result.score:
                    result = new_result

        # Return best result
        return max(results, key=lambda x: x.score)

    def search(self, query: str, return_all=False) -> dict:
        """Main search method using TTC"""
        best_result = self.beam_search_results(query)

        if return_all:
            return {
                "query": best_result.query,
                "urls": best_result.urls,
                "contents": best_result.contents,
                "score": best_result.score
            }

        # Return just the best content if return_all is False
        return {
            "content": " ".join(best_result.contents[:3]) if best_result.contents else "",
            "score": best_result.score
        }


# Example usage
if __name__ == "__main__":
    searcher = TTCSemanticSearch(n_candidates=4, beam_width=2)

    # Example search
    query = "What is the difference between Python and JavaScript?"
    results = searcher.search(query, return_all=True)

    print("Original Query:", query)
    print("\nBest Search Query Used:", results["query"])
    print("\nConfidence Score:", results["score"])
    print("\nTop URLs:")
    for url in results["urls"][:3]:
        print(f"- {url}")
    print("\nExtracted Content (First 200 chars):")
    if results["contents"]:
        print(results["contents"][0][:200] + "...")