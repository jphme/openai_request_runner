#TODO FIX

from request_runner import APIRequest, process_api_requests_from_list
from openaischema import OpenAISchema
from pydantic import BaseModel, Field
import json
import asyncio
from openai.openai_object import OpenAIObject
from typing import Any
import tiktoken
import logging
from dataclasses import dataclass

class LanguageCode(OpenAISchema):
    """"A single language code in ISO 639-1 format"""
    lc: str = Field(..., description="Language code (e.g. 'en', 'de', 'fr')")

class LanguageClassification(OpenAISchema):
    """Classify the languages of a user prompt."""
    
    language_codes: list[LanguageCode] = Field(default_factory=list, description="A list of up to 2 languages present in the text. Exclude code sections, loanwords and technical terms in the text when deciding on the language codes. You have to output at least one language code, even if you are not certain or the text is very short!", max_items=2)
    main_language_code: LanguageCode = Field(..., description="Main Language of the text.")

@dataclass
class LanguageClassificationRequest(APIRequest):

    def __post_init__(self):
        self.system_msg = "You are a world-class linguist and fluent in all major languages. Your job is to determine which languages are present in the user text and which one is the main language."
        self.functions = [LanguageClassification.openai_schema]
        self.function_call: dict|str = {"name": "LanguageClassification"}

    def preprocess_messages(self) -> list[dict]:
        #for sharegpt4 format:
        """[
        {
            "items": [
                {
                    "from": "human",
                    "value": "....in detail James Barr's boo...."
                },
                {
                    "from": "gpt",
                    "value": "8abf\u55ae\u8a5e\u7684\u5b57\....."
                }, .....
                ],
            "id": "Iaq2CFd",
            ...
        }
        ]
        """
        
        assert self.request_json['items'][0]['from']=="human"
        self.messages=[
            {
                "role": "system",
                    "content": self.system_msg,
            },
            {
                "role": "user",
                "content": f"{self.request_json['items'][0]['value'][:200]}",
            },
                ]
        return self.messages
    
    def postprocess_response(self, response: OpenAIObject) -> Any:
        #customize for results
        try:
            lang_class=LanguageClassification.from_response(response)
        except AttributeError as e:
            logging.warning(f"Could not classify languages for {self.task_id}, parsing error")
            raise e
        encoding = tiktoken.get_encoding(self.token_encoding_name)
        num_tokens=0
        for message in self.request_json['items']:
            try:
                num_tokens += len(encoding.encode(message['value']))
            except:
                logging.debug(f"Could not encode messages for {self.task_id}")
                continue
        res_dict={'num_languages':len(lang_class.language_codes),
                  'main_language':lang_class.main_language_code.lc,
                  'id':self.request_json['id'],
                  'task_id':self.task_id,
                  'turns': len(self.request_json['items']),
                  'tokens': num_tokens,}
        return res_dict
    
if __name__ == "__main__":
    openai_logger = logging.getLogger("openai")
    # Set the logging level for the logger to WARNING
    openai_logger.setLevel(logging.WARNING)
    with open("/Users/jph/dev2/german_finetuning_data/openchat/sharegpt_gpt4.json", "r") as f:
        sharegpt_gpt4_train = json.load(f)
    # run script
    status_tracker = asyncio.run(
        process_api_requests_from_list(
            input_list=sharegpt_gpt4_train, #[:1500],
            request_class=LanguageClassificationRequest,
            save_filepath="/Users/jph/dev2/german_finetuning_data/openchat/sharegpt_gpt4_language.jsonl",
            max_attempts=1
        )
    )
    print(status_tracker)