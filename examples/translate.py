# This example shows how to use the OpenAIRequestRunner to translate a conversation from English to German.
# see example input and out in example_input_sharegpt.json and example_output_translate.jsonl

from pydantic import Field
import json
import asyncio
from openai.openai_object import OpenAIObject
from typing import Any
import logging
from openai_request_runner import OpenAISchema, process_api_requests_from_list


# Classes for OpenAI Functions
class TranslatedTurn(OpenAISchema):
    """The correct translation of a conversation turn according to the instructions."""

    translation: str = Field(..., description="Translated text.")


class TranslatedConversation(OpenAISchema):
    """A list of multiple translations of conversation turns."""

    translated_turns: list[TranslatedTurn] = Field(
        default_factory=list,
        description="A list of translated conversation turns. Turns are the 'value' fields in the list.",
    )


# Template for system message which sets the context for the assistant.
system_template = """
You are a brilliant translator and your job is to translate the conversation into German. 

Please adhere to the following rules:
- Do not change the meaning of the text
... [snipped for brevity]
When in doubt don't translate word for word but rather translate the meaning of the sentence in a way that sounds natural in German!

The conversation is in a json template with a list of alternate turns between the user and the assistant.
"""


def preprocess_messages_sharegpt(request_json: dict, metadata: dict) -> list[dict]:
    """
    Preprocesses the given request JSON to extract the conversation and build the messages list for API request.

    Args:
    - request_json (dict): The input request containing the conversation.
    - metadata (dict): Metadata associated with the request.

    Returns:
    - list[dict]: A list containing the system and user messages to be sent to the API.
    """
    # Concatenate all messages into one string
    conversation_string = "\n---!!---\n".join(
        [item["value"] for item in request_json["items"]]
    )

    messages = [
        {
            "role": "system",
            "content": metadata["system_msg"],
        },
        {"role": "user", "content": json.dumps(request_json["items"])},
    ]
    return messages


def postprocess_response(
    response: OpenAIObject, request_json: dict, metadata: dict
) -> Any:
    """
    Postprocesses the API response to obtain translated conversation.

    Args:
    - response (OpenAIObject): The response object from the OpenAI API call.
    - request_json (dict): The original request sent to the API.
    - metadata (dict): Metadata associated with the API request.

    Returns:
    - dict: A dictionary containing the translated conversation and related information.
    """
    try:
        translated = TranslatedConversation.from_response(response)
    except AttributeError as e:
        logging.warning(
            f"Could not get translated text for {metadata['task_id']}, parsing error"
        )
        raise e

    res_dict = {
        "translated_text_turns": [
            item.translation for item in translated.translated_turns
        ],
        "id": request_json["id"],
        "task_id": metadata["task_id"],
        "turns": len(request_json["items"]),
    }
    return res_dict


# Setting up logging for OpenAI to suppress verbose logs
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)

# Load input data for processing
with open("examples/example_input_sharegpt.json", "r") as f:
    sharegpt_gpt4_train = json.load(f)

# Process the requests and obtain results
results = asyncio.run(
    process_api_requests_from_list(
        inputs=iter(sharegpt_gpt4_train),
        max_attempts=1,
        system_msg=system_template,
        preprocess_function=preprocess_messages_sharegpt,
        postprocess_function=postprocess_response,
        functions=[TranslatedConversation.openai_schema],
        function_call={"name": "TranslatedConversation"},
        save_filepath="examples/example_output_translate.jsonl",
        check_finished_ids=True,
        num_max_requests=2,
    )
)
