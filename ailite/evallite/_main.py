import json

from typing import Tuple, Optional, List, Any, Union, Type

from deepeval.metrics.answer_relevancy.schema import Statements, AnswerRelvancyVerdict, Verdicts, Reason
from pydantic import BaseModel

from ailitellm import HFModelType,ai
from deepeval.models import DeepEvalBaseLLM

class Steps(BaseModel):
    steps: List[str]

class ReasonScore(BaseModel):
    reason: str
    score: int


class DeepEvalLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[HFModelType] = "NousResearch/Hermes-3-Llama-3.1-8B",
        *args,
        **kwargs,
    ):
        self.model_name = model if model else "default"
        self.args = args
        self.kwargs = kwargs
        self.evaluation_cost = 0  # Initialize evaluation cost
        super().__init__(self.model_name)

    def load_model(self):
        return self.model_name

    def _parse_ai_response(self, response: str) -> dict:
        """Convert AI response to JSON format with better error handling"""
        try:
            # Remove markdown code block if present
            response = response.replace('```json', '').replace('```', '').strip()
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return self._create_structured_response(response)
            except:
                return self._create_structured_response(response)

    def _create_structured_response(self, response: str) -> dict:
        """Create structured response based on content type"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        # Try to identify steps format
        if any(line.startswith(str(i) + '.') for i in range(1, 10) for line in lines):
            steps = [line for line in lines if any(line.startswith(str(i) + '.') for i in range(1, 10))]
            return {"steps": steps}

        # Try to identify score/reason format
        import re
        score_match = re.search(r'\b([0-9]|10)\b', response)
        if score_match:
            score = int(score_match.group())
            reason = response.replace(score_match.group(), '').strip()
            return {"score": score, "reason": reason}

        return {"content": response}

    def _process_response(self, response: str, schema: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
        """Process response according to schema with improved handling"""
        parsed_response = self._parse_ai_response(response)
        if not schema:
            return parsed_response
        if schema.__name__== 'AnswerRelvancyVerdict':
            return AnswerRelvancyVerdict(**parsed_response)
        elif schema.__name__ == 'Verdicts':
            return Verdicts(**parsed_response)
        elif schema.__name__ == 'Reason':
            return Reason(**parsed_response)
        elif schema.__name__ == 'Statements':
            return Statements(statements=parsed_response['statements'])
        elif schema.__name__ == 'Steps':
            if isinstance(parsed_response, dict) and 'steps' in parsed_response:
                return Steps(steps=parsed_response['steps'])
            return Steps(steps=[str(response)])
        elif schema.__name__ == 'ReasonScore':
            if isinstance(parsed_response, dict) and 'score' in parsed_response and 'reason' in parsed_response:
                return ReasonScore(score=parsed_response['score'], reason=parsed_response['reason'])
            # Handle non-standard response format
            score = parsed_response.get('score', 5)
            reason = parsed_response.get('reason', str(response))
            return ReasonScore(score=score, reason=reason)
        return response

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **kwargs) -> Union[BaseModel, str]:
        """Synchronous generation with schema support"""
        response = ai(prompt, model=self.model_name).choices[0].message.content
        processed_response = self._process_response(response, schema)
        return processed_response

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **kwargs) -> Union[
        BaseModel, str]:
        """Asynchronous generation"""
        response = ai(prompt, model=self.model_name).choices[0].message.content
        processed_response = self._process_response(response, schema)
        print(processed_response,flush=True)
        return processed_response

    def get_model_name(self):
        return self.model_name