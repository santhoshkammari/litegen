import asyncio
import json
import os
from typing import List, Dict, Optional, Any, Generator, Union, Literal
import concurrent.futures
from functools import partial

import requests
from liteauto import google, parse
from liteauto.parselite import aparse
from pydantic import BaseModel, Field
from litegen import LLM


class SearchQuery(BaseModel):
    """A single search query to be sent to a search engine."""
    query: str = Field(..., description="The search query string")


class SearchResult(BaseModel):
    """Search result from a search engine."""
    title: str = Field(..., description="Title of the search result")
    description: str = Field(..., description="Snippet or summary of the search result")
    url: str = Field(..., description="URL of the search result")


class StepSearchQueries(BaseModel):
    """Collection of search queries for a single step."""
    queries: List[SearchQuery] = Field(..., description="List of search queries for this step")


class StepSummary(BaseModel):
    """Summary of search results for a step."""
    summary: str = Field(..., description="Summary of all search results for this step")
    is_satisfied: bool = Field(..., description="Whether this step is satisfied by the search results")
    gap_analysis: Optional[str] = Field(None, description="Analysis of gaps in the information if not satisfied")


class PlanStep(BaseModel):
    """A single step in the search plan."""
    step_number: int = Field(..., description="Number of this step in the plan")
    description: str = Field(..., description="Description of what this step aims to accomplish")
    search_iterations: List[List[SearchResult]] = Field(default_factory=list,
                                                        description="Search results from each iteration")
    summary: Optional[StepSummary] = Field(None, description="Final summary of this step after completion")
    query: Optional[str] = Field(None, description="The original user query this step belongs to")


class SearchPlan(BaseModel):
    """Complete plan for addressing a search query."""
    query: str = Field(..., description="The original user query")
    steps: List[PlanStep] = Field(..., description="Steps to address the query")


class FinalReport(BaseModel):
    """Final comprehensive report to be presented to the user."""
    original_query: str = Field(..., description="The original user query")
    comprehensive_answer: str = Field(..., description="Comprehensive answer to the user's query")
    step_summaries: List[str] = Field(..., description="Summaries of each step's findings")
    sources: List[str] = Field(..., description="Sources used in generating the answer")


class StreamMessage(BaseModel):
    """Message for streaming updates to the UI."""
    type: str = Field(..., description="Type of message (plan, step, query, result, summary, report)")
    content: Any = Field(..., description="Content of the message")


class SearchParameters(BaseModel):
    """Parameters for conducting a search operation."""
    max_urls: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of URLs to fetch per search query (1-10)"
    )
    max_iterations: int = Field(
        default=2,
        ge=2,
        le=10,
        description="Maximum number of iterations per step (1-10)"
    )
    num_gen_search_queries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of search queries to generate per step (1-5)"
    )
    delay_between_searches: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Delay between search requests in seconds (0.1-2.0)"
    )
    justification: str = Field(
        default="",
        description="Justification for the selected parameters"
    )

class LLMSearch:
    """LLM-based search tool that performs step-by-step research with parallel processing."""

    def __init__(self, llm=None, llm_threads=4, search_threads=8,enable_think_tag=False,
                 model_name = None,
                 search_provider:Literal['google','duckduckgo','weblair'] = None,
                 search_parallel:bool=True):
        """
        Initialize the search tool with LLM and thread pools.

        Args:
            llm: LLM instance to use
            llm_threads: Number of threads for LLM operations
            search_threads: Number of threads for search operations
        """
        self.search_parallel = search_parallel
        self.model = llm
        self.model_name = model_name
        self.enable_think_tag = enable_think_tag
        self.llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=llm_threads)
        self.search_executor = concurrent.futures.ThreadPoolExecutor(max_workers=search_threads)
        self.search_provider = search_provider or google

    def llm(self,system_prompt=None,prompt=None,response_format=None,model_name=None):
        if model_name is None:
            model_name = self.model_name or os.environ['OPENAI_MODEL_NAME']
        return self.model(system_prompt=system_prompt,prompt=prompt,response_format=response_format,model=model_name)

    def _get_optimal_parameters(self, user_query: str) -> SearchParameters:
        """
        Determine optimal search parameters based on the user query.

        Args:
            user_query: The user's search query

        Returns:
            SearchParameters: Optimized parameters for the search process
        """
        system_prompt = """
        You are an AI search optimization specialist. Your task is to analyze a user's search query and determine the optimal search parameters that will produce the most helpful and relevant results.

        For each search query, you need to determine:

        1. max_urls: How many URLs to fetch per search query (1-10)
           - For simple fact-checking queries: 1-2 URLs is sufficient
           - For general knowledge questions: 3-5 URLs provides good coverage
           - For complex research topics: 6-10 URLs ensures comprehensive information
           - for other than this topic decide by your own number

        2. max_iterations: Maximum number of search iterations per research step (2-10)
           - For straightforward queries: 2 iterations is usually enough
           - For questions requiring verification: 3-5 iterations helps confirm information
           - For little higher topics: 5 - 8 iterations allows for thorough investigation
           - For other than and if you feel like answer can't be find easity go beyond 8 and upto 10

        3. num_gen_search_queries: Number of search queries to generate per step (1-5)
           - For simple questions: 1-2 queries is sufficient
           - For moderate complexity: 3 queries provides good coverage
           - For complex or ambiguous topics: 4-5 queries explores different aspects

        4. delay_between_searches: Delay between search requests in seconds (0.1-2.0)
           - Balance between speed and avoiding rate limiting

        IMPORTANT CONSIDERATIONS:
        - If the user explicitly mentions "quick", "fast", or "simple", prioritize speed (lower values)
        - If the user mentions "thorough", "comprehensive", or "detailed", prioritize completeness (higher values)
        - For fact-checking or simple lookups, use minimal parameters
        - For complex research topics, academic subjects, or technical questions, use more comprehensive parameters
        - For questions involving current events, controversies, or multiple perspectives, use higher values to ensure balanced information

        Provide a brief justification for your parameter selections.
        """

        prompt = f"Analyze this search query and determine the optimal search parameters: '{user_query}'"

        try:
            # Call the LLM with the SearchParameters model as the response format
            parameters = self.llm(system_prompt=system_prompt, prompt=prompt, response_format=SearchParameters)

            # Apply reasonable constraints to ensure the values are within acceptable ranges
            parameters.max_urls = max(1, min(parameters.max_urls, 10))
            parameters.max_iterations = max(1, min(parameters.max_iterations, 5))
            parameters.num_gen_search_queries = max(1, min(parameters.num_gen_search_queries, 5))
            parameters.delay_between_searches = max(0.1, min(parameters.delay_between_searches, 2.0))

            return parameters
        except Exception as e:
            # Provide sensible defaults in case of an error
            print(f"Error determining optimal parameters: {e}")
            return SearchParameters(
                max_urls=3,
                max_iterations=2,
                num_gen_search_queries=3,
                delay_between_searches=0.5,
                justification="Using default parameters due to error in parameter optimization."
            )


    async def __call__(self, user_query: str, max_urls=1,
                       max_iterations=1
                       ) -> Generator[Union[StreamMessage, FinalReport], None, None]:
        """
        Execute the complete search process for a user query with streaming updates.

        Args:
            user_query: The user's search query
            max_urls: Maximum number of URLs to fetch per query

        Yields:
            StreamMessage objects containing progress updates
            FinalReport as the last yielded item
        """

        # Step 0: Get optimal parameters
        loop = asyncio.get_event_loop()
        parameters:SearchParameters = await loop.run_in_executor(
            self.llm_executor,
            self._get_optimal_parameters,
            user_query
        )

        # Step 1: Create a plan
        import datetime
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        user_query += f" \n for your information current time is :{formatted_time}"

        # Run plan creation in thread pool
        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(
            self.llm_executor,
            self._create_search_plan,
            user_query
        )

        if self.enable_think_tag:
            yield StreamMessage(type="plan",content="<think>")
            yield StreamMessage(type="parameters", content=f"Search parameters: {parameters.max_urls}, {parameters.max_iterations}, {parameters.num_gen_search_queries}, {parameters.delay_between_searches}")
            yield StreamMessage(type="parameters", content=f"Search parameters: {parameters.max_urls} urls, {parameters.max_iterations}iters, {parameters.num_gen_search_queries} queries per step, {parameters.delay_between_searches} delay.")
            yield StreamMessage(type="parameters", content=f"Using search parameters: {parameters.justification}\n")


        yield StreamMessage(type="plan", content=f"Created research plan with {len(plan.steps)} steps")

        for step in plan.steps:
            yield StreamMessage(type="plan_step", content=f"{step.step_number}. {step.description}")

        # Step 2: Execute each step in the plan
        for step in plan.steps:
            # Set the query context for each step
            step.query = plan.query
            yield StreamMessage(type="step", content=f'\nPerforming "{step.description}\n"')

            # Execute the step and yield progress updates
            async for update in self._execute_step(
                step,
                parameters.max_urls,
                parameters.max_iterations,
                parameters.num_gen_search_queries,
                parameters.delay_between_searches
            ):
                yield update

        # First show the progress message about generating the final report
        yield StreamMessage(type="progress", content=f"Total {len(plan.steps)} steps completed")

        sources = []

        for step in plan.steps:
            for iteration in step.search_iterations:
                for result in iteration:
                    if result.url not in sources:
                        sources.append(result.url)

        if self.enable_think_tag:
            yield StreamMessage(type="plan",content="</think>")

        yield StreamMessage(type="progress", content=f"Total {len(sources)} Sources ...")

        # Generate final report in thread pool
        loop = asyncio.get_event_loop()
        final_report = await loop.run_in_executor(
            self.llm_executor,
            self._generate_final_report,
            plan
        )

        yield final_report

    def _create_search_plan(self, user_query: str) -> SearchPlan:
        """Generate a step-by-step plan to address the user's query."""
        system_prompt = """
        You are a research planning assistant. Your task is to create a detailed, step-by-step plan to thoroughly research and answer the user's query.

        For a given query, create a series of logical steps that would lead to a comprehensive answer.
        Each step should:
        1. Focus on a specific aspect of the overall query
        2. Build upon information that would be gathered in previous steps
        3. Be concrete and specific enough that it can be researched with search engine queries
        4. Be between 3-5 steps total, focusing on the most important aspects
        5. Each step should address distinct aspects of the query

        Your output should follow the SearchPlan schema with appropriate steps.
        """

        prompt = f"Create a research plan for thoroughly answering this query: {user_query}"

        try:
            # Define the SearchPlan model as the response format
            plan = self.llm(system_prompt=system_prompt, prompt=prompt, response_format=SearchPlan)
            return plan
        except Exception as e:
            # Create a fallback plan with basic steps
            return SearchPlan(
                query=user_query,
                steps=[
                    PlanStep(step_number=1, description="Understand the basic concepts related to the query"),
                    PlanStep(step_number=2, description="Identify specific requirements or constraints"),
                    PlanStep(step_number=3, description="Research available options and solutions"),
                    PlanStep(step_number=4, description="Compare alternatives based on key criteria"),
                    PlanStep(step_number=5, description="Formulate recommendations based on findings")
                ]
            )

    def _generate_search_queries(self, step: PlanStep, n: int) -> List[SearchQuery]:
        """Generate search queries for a given step."""
        system_prompt = f"""
        You are a search query formulation specialist. Your task is to generate effective search queries 
        that will help answer a specific research step.

        Given a research step and the overall query, generate {n} different search queries that:
        1. Are specific to the step's goal and will return relevant results
        2. Use varied approaches to find complementary information
        3. Include relevant keywords related to the specific step
        4. Use search operators effectively where appropriate (quotes for exact phrases, etc.)
        5. Are formulated to fill any gaps identified in previous iterations

        IMPORTANT: 
        - Each query must be a concrete, well-formed search query (not placeholders with [brackets])
        - DO NOT leave any queries empty or use template placeholders
        - Make each query specific and actionable
        - If you have information from previous iterations, focus on addressing the gaps

        Your output should follow the StepSearchQueries schema with {n} complete, ready-to-use queries.
        """

        previous_info = ""
        if step.search_iterations:
            previous_info = "\n\nBased on previous searches, we've learned:\n"
            for i, iteration in enumerate(step.search_iterations):
                if iteration:  # Only include if there are actual results
                    previous_info += f"Iteration {i + 1}:\n"
                    for result in iteration[:3]:  # Limit to 3 results per iteration to avoid overloading
                        previous_info += f"- {result.title}: {result.description[:100]}...\n"

            if step.summary:
                previous_info += f"\nCurrent gaps in knowledge: {step.summary.gap_analysis}\n"

        prompt = f"""
        Generate {n} effective search queries for the following research step:

        Step {step.step_number}: {step.description}

        This is part of researching the overall query: "{step.query}"
        {previous_info}

        Create {n} concrete, specific search queries that will directly help with this step.
        """

        try:
            queries_response = self.llm(system_prompt=system_prompt, prompt=prompt, response_format=StepSearchQueries)
            # Validate that we have actual queries
            valid_queries = [q for q in queries_response.queries if q.query and q.query.strip()]

            # If we don't have enough valid queries, generate some basic ones
            if len(valid_queries) < n:
                terms = step.description.lower().split()
                keywords = [w for w in terms if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'about']]

                if step.query:
                    fallback_queries = [
                        SearchQuery(query=f"{step.query} {step.description}"),
                        SearchQuery(query=f"how to {step.description}"),
                        SearchQuery(query=f"best {keywords[0] if keywords else ''} for {step.query}")
                    ]
                    valid_queries.extend(fallback_queries)

            return valid_queries[:n]  # Return at most n queries
        except Exception as e:
            return [SearchQuery(query=f"{step.query} {step.description}")]

    def _perform_web_search_sync(self, query: str, max_urls) -> List[SearchResult]:
        """Synchronous version of web search for thread pool execution."""
        try:
            res = self.search_provider(query, max_urls=max_urls, advanced=True)
            titles_map = {x.url: x.title for x in res}

            try:
                # Note: This is synchronous parsing, which is fine for thread pool
                response = parse([r.url for r in res])
                return [
                    SearchResult(
                        title=titles_map.get(r.url, "Untitled"),
                        description=str(r.content)[:800],  # Limit description length
                        url=r.url
                    ) for r in response if hasattr(r, 'content') and r.content
                ]
            except Exception as parse_error:
                # Return basic results if parsing fails
                return [
                    SearchResult(
                        title=r.title if hasattr(r, 'title') else "Untitled",
                        description=r.snippet[:500] if hasattr(r, 'snippet') else "No description available",
                        url=r.url
                    ) for r in res[:max_urls]
                ]
        except Exception as e:
            # Return empty list on complete failure
            return []

    async def _perform_web_search(self, query: str, max_urls) -> List[SearchResult]:
        """Perform a web search using the specified query and return results."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.search_executor,
            partial(self._perform_web_search_sync, query, max_urls)
        )

    def _summarize_search_results(self, step: PlanStep, latest_results: List[SearchResult]) -> StepSummary:
        """Summarize search results and determine if the step is satisfied."""
        # Get all results so far
        all_results = []
        for iteration in step.search_iterations:
            all_results.extend(iteration)

        # Add latest results
        all_results.extend(latest_results)

        # Limit the number of results to prevent token overflow
        if len(all_results) > 8:
            # Prioritize latest results and select representative ones from earlier iterations
            selected_results = latest_results + all_results[:-len(latest_results)]
            selected_results = selected_results[:8]
        else:
            selected_results = all_results

        # Format result text
        result_text = "\n\n".join([
            f"Title: {result.title}\nSnippet: {result.description[:250]}...\nURL: {result.url}"
            for result in selected_results
        ])

        system_prompt = """
        You are a research analysis assistant. Your task is to analyze search results for a specific research step 
        and determine if they satisfy the information needs of that step.

        Analyze the search results and:
        1. Create a comprehensive summary of the key information found (max 200 words)
        2. Determine if the information is sufficient to satisfy the research step (yes/no)
        3. If not satisfied, provide a specific analysis of what information is still missing

        Be critical in your assessment - only mark a step as satisfied if it truly provides enough 
        information to answer that aspect of the user's query.

        Your output should follow the StepSummary schema.
        """

        prompt = f"""
        Analyze the following search results for research step {step.step_number}: {step.description}

        This step is part of researching the overall query: "{step.query}"

        Search Results:
        {result_text}

        Based on these results, provide a concise summary and determine if this step is satisfied.
        If it's not satisfied, explain specifically what information is still missing.
        """

        try:
            summary = self.llm(system_prompt=system_prompt, prompt=prompt, response_format=StepSummary)
            return summary
        except Exception as e:
            # Create a fallback summary
            return StepSummary(
                summary=f"Found {len(all_results)} search results related to {step.description}.",
                is_satisfied=len(all_results) > 2,  # Assume satisfied if we have at least 3 results
                gap_analysis="Unable to perform detailed gap analysis due to an error."
            )

    async def _execute_step(self, step: PlanStep, max_urls, max_iterations,
                            num_gen_search_queries,
                            delay_between_searches) -> Generator[StreamMessage, None, None]:
        """Execute a single step in the search plan."""
        # Limit the number of search iterations per step

        for iteration in range(max_iterations):
            # Generate queries in thread pool
            loop = asyncio.get_event_loop()
            queries = await loop.run_in_executor(
                self.llm_executor,
                partial(self._generate_search_queries, step, num_gen_search_queries)
            )


            # Execute searches in parallel and collect results
            iteration_results = []
            search_tasks = []

            if self.search_parallel:
                for idx, query in enumerate(queries, 1):
                    yield StreamMessage(type="progress", content=f"Searching {query.query}")
                    search_tasks.append(self._perform_web_search(query.query, max_urls=max_urls))

                # Gather all search results
                search_results = await asyncio.gather(*search_tasks)
            else:
                search_results = []
                for idx, query in enumerate(queries, 1):
                    yield StreamMessage(type="progress", content=f"Searching {query.query}")
                    await asyncio.sleep(delay_between_searches)  # 1 second between search requests
                    results = await self._perform_web_search(query.query, max_urls=max_urls)
                    search_results.append(results)



            for i, results in enumerate(search_results):
                iteration_results.extend(results)

            # Add results to the step's iterations
            step.search_iterations.append(iteration_results)

            # Summarize in thread pool
            summary = await loop.run_in_executor(
                self.llm_executor,
                partial(self._summarize_search_results, step, iteration_results)
            )

            step.summary = summary  # Always update the summary

            yield StreamMessage(type="summary",
                                content=f"\n{summary.summary[:100]}...")

            if summary.is_satisfied:
                yield StreamMessage(type="status",
                                    content=f"Step {step.step_number} satisfied after iteration {iteration + 1}")
                break
            else:
                yield StreamMessage(type="status",
                                    content=f"\n Missing {summary.gap_analysis}")
                if iteration + 1 < max_iterations:
                    yield StreamMessage(type="progress", content="Trying additional queries...")
                else:
                    break

        # Ensure the step has a summary even if all iterations failed
        if not step.summary:
            step.summary = StepSummary(
                summary=f"Limited information found for this step.",
                is_satisfied=False,
                gap_analysis="Unable to gather sufficient information for this step."
            )
            yield StreamMessage(type="warning",
                                content="No summary could be generated for this step.")

    def _generate_final_report(self, plan: SearchPlan) -> StreamMessage:
        """Generate a final report based on all completed steps."""
        # Prepare a summary of all steps
        steps_summary = []
        sources = []

        for step in plan.steps:
            if step.summary:
                summary_text = f"Step {step.step_number}: {step.description}\n{step.summary.summary}"
                steps_summary.append(summary_text)

                for iteration in step.search_iterations:
                    for result in iteration:
                        if result.url not in sources:
                            sources.append(result.url)

        system_prompt = """
        You are a research synthesis specialist. Your task is to create a comprehensive, well-structured final report 
        that directly addresses the user's original query based on the research conducted.

        Create a report that:
        1. Directly answers the user's original query with actionable information
        2. Synthesizes information from all research steps into a coherent narrative
        3. Is well-structured with clear sections and logical flow
        4. Uses concise, clear language focused on delivering valuable insights
        5. Highlights key findings and recommendations
        6. Acknowledges any limitations in the research

        Focus on being accurate and factual - do not make up information that wasn't found in the research.
        If information is limited, acknowledge the limitations rather than filling in gaps with speculation.
        """

        # Combine step summaries for the prompt
        combined_summaries = "\n\n".join(steps_summary)

        prompt = f"""
        Create a comprehensive final report for the user's query:

        Original Query: {plan.query}

        Research Findings from Each Step:
        {combined_summaries}

        Based on the results and findings, generate an extensive report in a neat, formatted style.
        """

        # Generate the report content
        report_content = self.llm(system_prompt=system_prompt, prompt=prompt, response_format=None)

        # Also include sources at the end of the report
        sources_text = "\n\nSources:\n" + "\n".join([f"- {x}" for x in sources])
        final_report = report_content + sources_text

        if "</think>" in final_report:
            final_report = final_report[final_report.index("</think>") + len("</think>"):]

        # Return the final report as a StreamMessage
        return StreamMessage(type="final_report", content=final_report)


async def main(user_query):
    # Create LLMSearch with specified thread pool sizes
    search_tool = LLMSearch(LLM('dsollama'), llm_threads=4, search_threads=8)
    async for update in search_tool(user_query):
        if isinstance(update, StreamMessage):
            print(f"[{update.type}] {update.content}")
        elif isinstance(update, FinalReport):
            print(update.comprehensive_answer)


if __name__ == "__main__":
    asyncio.run(main("find best paper related to genai"))