from typing import Any
import json


def load_jsonl(filename: str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    with open(filename, "r") as f:
        erg = [json.loads(item) for item in f.readlines()]
    return erg


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")
