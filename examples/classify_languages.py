from pydantic import Field
import json
import asyncio
from openai.openai_object import OpenAIObject
from typing import Any
import tiktoken
import logging
from openai_request_runner import OpenAISchema, process_api_requests_from_list


# Classes for OpenAI Functions
class LanguageCode(OpenAISchema):
    """ "A single language code in ISO 639-1 format"""

    lc: str = Field(..., description="Language code (e.g. 'en', 'de', 'fr')")


class LanguageClassification(OpenAISchema):
    """Classify the languages of a user prompt."""

    language_codes: list[LanguageCode] = Field(
        default_factory=list,
        description="A list of up to 2 languages present in the text. Exclude code sections, loanwords and technical terms in the text when deciding on the language codes. You have to output at least one language code, even if you are not certain or the text is very short!",
        max_items=2,
    )
    main_language_code: LanguageCode = Field(
        ..., description="Main Language of the text."
    )


# Function for postprocessing the response


def postprocess_response(
    response: OpenAIObject, request_json: dict, metadata: dict
) -> Any:
    # customize for results
    try:
        lang_class = LanguageClassification.from_response(response)
    except AttributeError as e:
        logging.warning(
            f"Could not classify languages for {metadata['task_id']}, parsing error"
        )
        raise e
    encoding = tiktoken.get_encoding(metadata["token_encoding_name"])
    num_tokens = 0
    for message in request_json["items"]:
        try:
            num_tokens += len(encoding.encode(message["value"]))
        except:
            logging.debug(f"Could not encode messages for {metadata['task_id']}")
            continue
    res_dict = {
        "num_languages": len(lang_class.language_codes),
        "main_language": lang_class.main_language_code.lc,
        "language_codes": [item.lc for item in lang_class.language_codes],
        "id": request_json["id"],
        "task_id": metadata["task_id"],
        "turns": len(request_json["items"]),
        "tokens": num_tokens,
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
        system_msg="You are a world-class linguist and fluent in all major languages. Your job is to determine which languages are present in the user text and which one is the main language.",
        postprocess_function=postprocess_response,
        functions=[LanguageClassification.openai_schema],
        function_call={"name": "LanguageClassification"},
        save_filepath="examples/example_output_lc.jsonl",
        check_finished_ids=True,
        num_max_requests=2,
    )
)
