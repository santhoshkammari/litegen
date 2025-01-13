import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from functools import wraps
import threading
from typing import Optional, Dict, Any
from openai.types.chat import ChatCompletion


class TraceLLM:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TraceLLM, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.build_storage_dir()
            self.current_experiment = "default"
            self._local = threading.local()
            self.initialized = True

    def _serialize_response(self, response):
        """Convert OpenAI response to serializable format"""
        if isinstance(response, ChatCompletion):
            return {
                "id": response.id,
                "model": response.model,
                "created": response.created,
                "choices": [{
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                } for choice in response.choices],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "total_tokens": response.usage.total_tokens if response.usage else None
                } if response.usage else None
            }
        return str(response)

    def openai_autolog(self):
        """Enable automatic logging for OpenAI-style APIs"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                trace_id = kwargs.pop('trace_id',None) or str(uuid.uuid4())
                # Extract messages from kwargs or args
                messages = kwargs.get('messages', args[0] if args else [])
                trace_extra_info = kwargs.pop('trace_extra_info',{})

                trace = {
                    "trace_id": trace_id,
                    "experiment": self.current_experiment,
                    "timestamp": datetime.now().isoformat(),
                    "inputs": {
                        "messages": messages,
                        "model": kwargs.get('model', 'unknown')
                    },
                    "outputs": None,
                    "status": None,
                    "duration": None,
                    "trace_extra_info":trace_extra_info
                }

                start_time = time.time()
                try:
                    response = await func(*args, **kwargs)
                    trace["outputs"] = {
                        "response": self._serialize_response(response),
                        "content": response.choices[0].message.content if response.choices else None
                    }
                    trace["status"] = "success"
                except Exception as e:
                    trace["error"] = str(e)
                    trace["status"] = "error"
                    raise
                finally:
                    trace["duration"] = time.time() - start_time
                    self._save_trace(trace)
                return response

            return wrapper

        return decorator

    def set_experiment(self, experiment_name: str):
        """Set the current experiment name"""
        if os.environ.get("OPENAI_TRACING","true") == "true":
            self.current_experiment = experiment_name
            exp_dir = self.storage_dir / experiment_name
            exp_dir.mkdir(exist_ok=True)

    def _save_trace(self, trace: Dict[str, Any]):
        """Save trace to storage with safe file writing"""
        exp_dir = self.storage_dir / self.current_experiment
        trace_file = exp_dir / f"{trace['trace_id']}.json"

        # Write to a temporary file first
        temp_file = trace_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(trace, f, indent=2)
                f.flush()  # Ensure all data is written

            # Rename temp file to final file (atomic operation)
            temp_file.replace(trace_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()  # Clean up temp file if something went wrong
            raise e

    def get_experiments(self):
        """Get list of all experiments"""
        return [d.name for d in self.storage_dir.iterdir() if d.is_dir()]

    def get_traces(self, experiment: Optional[str] = None):
        """Get all traces for an experiment"""
        exp_dir = self.storage_dir / (experiment or self.current_experiment)
        traces = []
        for trace_file in exp_dir.glob("*.json"):
            with open(trace_file) as f:
                traces.append(json.load(f))
        return traces

    def cleanup_incomplete_traces(self):
        """Clean up any incomplete trace files"""
        for exp_dir in self.storage_dir.iterdir():
            if exp_dir.is_dir():
                # Clean up temp files
                for temp_file in exp_dir.glob('*.tmp'):
                    temp_file.unlink()

                # Clean up incomplete JSON files
                for trace_file in exp_dir.glob('*.json'):
                    try:
                        with open(trace_file) as f:
                            json.load(f)  # Try to parse JSON
                    except json.JSONDecodeError:
                        trace_file.unlink()  # Delete corrupted file

    def build_storage_dir(self):
        if os.environ.get("OPENAI_TRACING","true") == "true":
            self.storage_dir = Path("traces")
            self.storage_dir.mkdir(exist_ok=True)