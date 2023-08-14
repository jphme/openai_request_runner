# copied from https://github.com/jxnl/openai_function_call

import json
from pydantic import BaseModel


def _remove_a_key(d, remove_key) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


class OpenAISchema(BaseModel):
    @classmethod
    @property
    def openai_schema(cls):
        schema = cls.schema()
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        parameters["required"] = sorted(parameters["properties"])
        _remove_a_key(parameters, "title")
        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    @classmethod
    def from_response(cls, completion, throw_error=True):
        message = completion.choices[0].message

        if throw_error:
            assert "function_call" in message, "No function call detected"
            assert (
                message["function_call"]["name"] == cls.openai_schema["name"]
            ), "Function name does not match"

        function_call = message["function_call"]
        arguments = json.loads(function_call["arguments"])
        return cls(**arguments)

    @classmethod
    def from_response_langchain(cls, completion):
        function_call = completion.generations[0][0].message.additional_kwargs[
            "function_call"
        ]
        arguments = json.loads(function_call["arguments"])
        return cls(**arguments)
