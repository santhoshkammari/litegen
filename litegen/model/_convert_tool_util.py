from typing import Any, Callable, Optional, Union, Dict, Type

from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from pydantic import BaseModel

from langchain_core.utils.function_calling import convert_to_openai_function

class FormatConverter:
    """A flexible converter class to handle different function/tool formats."""

    def __init__(self, strict: Optional[bool] = None):
        self.strict = strict

    def convert(
        self,
        source: Union[Dict[str, Any], Type[BaseModel], Callable],
        target_format: str = "openai",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Convert a source format to target format with extensible options.

        Args:
            source: The source object to convert. Can be:
                - Dictionary (OpenAI function, JSON schema, Anthropic tool, Amazon Bedrock tool)
                - Pydantic BaseModel class
                - TypedDict class
                - Python function
            target_format: The desired output format (default: "openai")
            **kwargs: Additional conversion parameters for future extensibility

        Returns:
            Dict[str, Any]: Converted format matching the target specification

        Raises:
            ValueError: If source format is unsupported or conversion fails
        """
        # First detect the source format
        source_format = self._detect_format(source)

        # Convert to intermediate format
        intermediate = self._to_intermediate(source, source_format)

        # Apply any format-specific transformations
        result = self._to_target_format(intermediate, target_format)

        # Apply strict mode if specified
        if self.strict is not None:
            result = self._apply_strict_mode(result)

        return result

    def _detect_format(
        self,
        source: Union[Dict[str, Any], Type[BaseModel], Callable]
    ) -> str:
        """Detect the format of the source object."""
        if isinstance(source, dict):
            if "toolSpec" in source:
                return "amazon_bedrock"
            elif all(k in source for k in ("name", "input_schema")):
                return "anthropic"
            elif "name" in source:
                return "openai"
            elif "title" in source:
                return "json_schema"
        elif isinstance(source, type):
            if is_basemodel_subclass(source):
                return "pydantic"
            elif is_typeddict(source):
                return "typeddict"
        elif callable(source):
            return "python_function"

        raise ValueError(
            f"Unsupported format: {source}. Must be a Dict, pydantic.BaseModel, "
            "TypedDict, or Callable."
        )

    def _to_intermediate(
        self,
        source: Union[Dict[str, Any], Type[BaseModel], Callable],
        source_format: str
    ) -> Dict[str, Any]:
        """Convert source to intermediate format."""
        converters = {
            "amazon_bedrock": self._from_bedrock,
            "anthropic": self._from_anthropic,
            "openai": self._from_openai,
            "json_schema": self._from_json_schema,
            "pydantic": self._from_pydantic,
            "typeddict": self._from_typeddict,
            "python_function": self._from_python_function
        }

        converter = converters.get(source_format)
        if not converter:
            raise ValueError(f"Unsupported source format: {source_format}")

        return converter(source)

    def _to_target_format(
        self,
        intermediate: Dict[str, Any],
        target_format: str
    ) -> Dict[str, Any]:
        """Convert intermediate format to target format."""
        if target_format == "openai":
            return self._to_openai(intermediate)
        elif target_format == "openai_tool":
            return {
                "type": "function",
                "function": self._to_openai(intermediate)
            }
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

    def _apply_strict_mode(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strict mode settings to the result."""
        if "strict" in result and result["strict"] != self.strict:
            raise ValueError(
                f"Result already has 'strict' value {result['strict']} "
                f"which differs from specified {self.strict}"
            )

        result["strict"] = self.strict
        if self.strict:
            if "parameters" in result:
                result["parameters"] = self._recursive_set_additional_properties_false(
                    result["parameters"]
                )
        return result

    # Source format converters
    def _from_bedrock(self, source: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "name": source["toolSpec"]["name"],
            "parameters": source["toolSpec"]["inputSchema"]["json"]
        }
        if "description" in source["toolSpec"]:
            result["description"] = source["toolSpec"]["description"]
        return result

    def _from_anthropic(self, source: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "name": source["name"],
            "parameters": source["input_schema"]
        }
        if "description" in source:
            result["description"] = source["description"]
        return result

    def _from_openai(self, source: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v
            for k, v in source.items()
            if k in ("name", "description", "parameters", "strict")
        }

    def _from_json_schema(self, source: Dict[str, Any]) -> Dict[str, Any]:
        source_copy = source.copy()
        result = {"name": source_copy.pop("title")}
        if "description" in source_copy:
            result["description"] = source_copy.pop("description")
        if source_copy and "properties" in source_copy:
            result["parameters"] = source_copy
        return result

    def _from_pydantic(self, source: Type[BaseModel]) -> Dict[str, Any]:
        return convert_pydantic_to_openai_function(source)

    def _from_typeddict(self, source: Type) -> Dict[str, Any]:
        return _convert_typed_dict_to_openai_function(source)

    def _from_python_function(self, source: Callable) -> Dict[str, Any]:
        return convert_python_function_to_openai_function(source)

    def _to_openai(self, intermediate: Dict[str, Any]) -> Dict[str, Any]:
        """Convert intermediate format to OpenAI format."""
        return intermediate

    def _recursive_set_additional_properties_false(
        self,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively set additionalProperties to false in JSON schema."""
        result = schema.copy()
        result["additionalProperties"] = False

        if "properties" in result:
            for prop in result["properties"].values():
                if isinstance(prop, dict):
                    self._recursive_set_additional_properties_false(prop)

        return result


def is_basemodel_subclass(cls: Type) -> bool:
    """Check if a class is a subclass of Pydantic BaseModel."""
    return isinstance(cls, type) and issubclass(cls, BaseModel)


def is_typeddict(cls: Type) -> bool:
    """Check if a class is a TypedDict."""
    return hasattr(cls, "__annotations__") and hasattr(cls, "__total__")