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


system_template = """You are a brilliant translator and your job is to translate the conversation into German. 

Please adhere to the following rules:
- Do not change the meaning of the text
- Do not add or remove information
- Do not change the style of the text
- Do not change the tone of the text
- Do only translate english text to German and leave all other text as is
- Do not translate any names
- Use always the direct/casual form of German address ("Du" instead of "Sie") except in formal settings (e.g. if the user asks to write an official letter)
- Do not translate any computer code (most often denoted with triple backticks like ```) and comments in the code
- When in doubt don't translate word for word but rather translate the meaning of the sentence in a way that sounds natural in German!

The conversation is in a json template with a list of alternate turns between the user and the assistant.
"""


# Function for pre- and post-processing the response
def preprocess_messages_sharegpt(request_json: dict, metadata: dict) -> list[dict]:
    # concat all messages into one string
    conversation_string = "\n---!!---\n".join(
        [item["value"] for item in request_json["items"]]
    )

    messages = [
        {
            "role": "system",
            "content": metadata["system_msg"],
        },
        {
            "role": "user",
            "content": json.dumps(
                request_json["items"]
            ),  # conversation_string #"Translate ALL text below according to the instructions:\n-----\n"+
        },
    ]
    return messages


def postprocess_response(
    response: OpenAIObject, request_json: dict, metadata: dict
) -> Any:
    # customize for results
    try:
        translated = TranslatedConversation.from_response(response)
    except AttributeError as e:
        logging.warning(
            f"Could not get translated text for {metadata['task_id']}, parsing error"
        )
        raise e

    print(response)  # TODO DEBUG RAus
    res_dict = {
        "translated_text_turns": [
            item.translation for item in translated.translated_turns
        ],
        "id": request_json["id"],
        "task_id": metadata["task_id"],
        "turns": len(request_json["items"]),
    }
    return res_dict


# RUN THE REQUESTS

openai_logger = logging.getLogger("openai")
# Set the logging level for the logger to WARNING
openai_logger.setLevel(logging.WARNING)
with open("examples/example_input_sharegpt.json", "r") as f:
    sharegpt_gpt4_train = json.load(f)

results = asyncio.run(
    process_api_requests_from_list(
        inputs=iter(sharegpt_gpt4_train),
        max_attempts=1,
        # model='gpt-4-0613',
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
