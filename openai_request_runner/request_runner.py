# from https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
from abc import abstractmethod
from ast import Call
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata
import openai
from openai.openai_object import OpenAIObject
from typing import Any, Callable
from pydantic import BaseModel, Field
from openaischema import OpenAISchema
from utils import append_to_jsonl

# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = (
        0  # used to cool off after hitting rate limits
    )
    num_prompt_tokens_used: int = 0
    num_completion_tokens_used: int = 0


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int | str
    request_json: dict
    attempts_left: int
    metadata: dict
    preprocess_messages: Callable[[dict, dict], list[dict]]
    postprocess_response: Callable[[OpenAIObject, dict, dict], dict]
    messages: list[dict] | None = None
    result: list = field(default_factory=list)
    return_results: bool = False

    def __post_init__(self):
        # load metadata from metadata object for more readable code
        self.token_encoding_name = self.metadata["token_encoding_name"]
        self.model = self.metadata["model"]
        self.system_msg = self.metadata["system_msg"]
        self.max_tokens = self.metadata["max_tokens"]
        self.functions = self.metadata["functions"]
        self.function_call = self.metadata["function_call"]
        self.metadata["task_id"] = self.task_id

    def num_tokens_consumed_from_request(self):
        """Count the number of tokens in the request. Only supports completion and embedding requests."""
        encoding = tiktoken.get_encoding(self.token_encoding_name)
        # if completions request, tokens = prompt + n * max_tokens

        num_tokens = 0
        self.messages = self.messages or self.preprocess_messages(
            self.request_json, self.metadata
        )
        for message in self.messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens -= 1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        # siehe https://github.com/openai/openai-cookbook/blob/4fd2b1a6d29d76dcdb3ae65ac12b1a71253d65b6/examples/api_request_parallel_processor.py#L348 ?
        return num_tokens + self.max_tokens

    async def call_api(
        self,
        retry_queue: asyncio.Queue,
        save_filepath: str | None,
        error_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        error_filepath = (
            ".".join(save_filepath.split(".")[:-1]) + "_errors.jsonl"
            if save_filepath
            else error_filepath
        )
        self.messages = self.messages or self.preprocess_messages(
            self.request_json, self.system_msg
        )
        params = dict(
            model=self.model,
            temperature=0,
            messages=self.messages,
            max_tokens=self.max_tokens,
            stream=False,
        )
        if self.functions is not None:
            params["functions"] = self.functions
            params["function_call"] = self.function_call  # type: ignore
        self.metadata.update(
            {key: value for key, value in params.items() if key != "messages"}
        )
        try:
            response = await openai.ChatCompletion.acreate(**params)
            assert isinstance(response, OpenAIObject)
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )
            status_tracker.num_prompt_tokens_used += response.usage.prompt_tokens
            status_tracker.num_completion_tokens_used += (
                response.usage.completion_tokens
            )
            response = self.postprocess_response(
                response, self.request_json, self.metadata
            )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
            # raise e #todo debug raus
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.task_id} failed after all attempts. Saving errors."
                )
                data = (
                    [self.messages, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.messages, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, error_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = [self.messages, response, self.metadata]  # type: ignore

            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

            if not save_filepath:
                return data
            else:
                logging.debug(f"Request {self.task_id} saved to {save_filepath}")
                append_to_jsonl(data, save_filepath)
                return None


def preprocess_messages_sharegpt(request_json: dict, metadata: dict) -> list[dict]:
    # example for first 200 chars from user input in sharegpt4 format:

    assert request_json["items"][0]["from"] == "human"
    messages = [
        {
            "role": "system",
            "content": metadata["system_msg"],
        },
        {
            "role": "user",
            "content": request_json["items"][0]["value"][:200],
        },
    ]
    return messages


def postprocess_response_default(
    response: OpenAIObject, request_json: dict, metadata: dict
) -> dict:
    # customize for results
    return dict(response.choices[0].message)


def get_finished_tasks_from_file(file_path: str) -> set[int]:
    """Get a list of task IDs that have already been completed."""
    try:
        with open(file_path) as file:
            # `requests` will provide requests one at a time
            finished = file.__iter__()
            finished_ids = set(
                int(list(json.loads(line).keys())[0]) for line in finished
            )
            logging.debug(f"Finished Tasks found and loaded")
            return finished_ids
    except FileNotFoundError:
        logging.debug(f"No finished tasks found")
        return set()


def get_id_from_finished_default(result_list: list) -> int:
    return result_list[2]["task_id"]


async def process_api_requests_from_list(
    input_list: list,
    save_filepath: str | None = None,
    error_filepath: str = "error_log.jsonl",
    token_encoding_name: str = "cl100k_base",
    model: str = "gpt-3.5-turbo-0613",
    system_msg: str = "You are a helpful assistant.",
    max_tokens: int = 200,
    id_field_getter: Callable[[dict], str | int] | None = lambda x: x["id"],
    functions: list | None = None,
    function_call: dict | str = "auto",
    preprocess_function: Callable[
        [dict, dict], list[dict]
    ] = preprocess_messages_sharegpt,
    postprocess_function: Callable[
        [OpenAIObject, dict, dict], dict
    ] = postprocess_response_default,
    check_finished_ids: bool = False,
    get_id_from_finished: Callable[[list], int | str] = get_id_from_finished_default,
    finished_ids: set[int] | None = None,
    max_requests_per_minute: float = 3000.0,
    max_tokens_per_minute: float = 85000.0,
    max_attempts: int = 3,
    logging_level: int = 10,
    verbose: bool = True,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    if save_filepath is None:
        write_to_file = False
    else:
        write_to_file = True

    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # initialize trackers
    results_future_list = []
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 1, 2, 3, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    if check_finished_ids:
        if save_filepath:
            try:
                with open(save_filepath) as file:
                    finished = file.__iter__()
                    finished_tasks = set(
                        [get_id_from_finished(json.loads(line)) for line in finished]
                    )
                    logging.debug(
                        f"{len(finished_tasks)} finished tasks found and loaded"
                    )
            except FileNotFoundError:
                logging.debug(f"No finished tasks found")
                finished_tasks = set()
        else:
            assert (
                finished_ids is not None
            ), "If check_finished_ids is True and not write to file, finished_ids must be provided"
            finished_tasks = finished_ids
    else:
        finished_tasks = set()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    task_generator = iter(input_list)
    logging.debug("Entering main loop")

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}")
            elif file_not_finished:
                try:
                    # get new request
                    next_task = next(task_generator)
                    if id_field_getter is None:
                        next_task_id = next(task_id_generator)
                    else:
                        next_task_id = id_field_getter(next_task)
                    if next_task_id in finished_tasks:
                        logging.debug(
                            f"Skipping request {next_task_id} because it is already finished"
                        )
                        continue
                    metadata = next_task.pop("metadata", {})
                    metadata["token_encoding_name"] = token_encoding_name
                    metadata["model"] = model
                    metadata["system_msg"] = system_msg
                    metadata["max_tokens"] = max_tokens
                    metadata["functions"] = functions
                    metadata["function_call"] = function_call

                    next_request = APIRequest(
                        task_id=next_task_id,
                        request_json=next_task,
                        attempts_left=max_attempts,
                        metadata=metadata,
                        preprocess_messages=preprocess_function,
                        postprocess_response=postprocess_function,
                        return_results=not write_to_file,
                    )
                    status_tracker.num_tasks_started += 1
                    if status_tracker.num_tasks_started % 200 == 0:
                        print(status_tracker)
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Read file exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity
            + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.num_tokens_consumed_from_request()
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                task = asyncio.create_task(
                    next_request.call_api(
                        retry_queue=queue_of_requests_to_retry,
                        save_filepath=save_filepath,
                        error_filepath=error_filepath,
                        status_tracker=status_tracker,
                    )
                )
                if not write_to_file:
                    results_future_list.append(task)
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (
            time.time() - status_tracker.time_of_last_rate_limit_error
        )
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (
                seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
            )
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )

    # after finishing, log final status
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )
    if verbose:
        print(status_tracker)
    if not write_to_file:
        logging.info(f"""Parallel processing complete. Results returned""")
        return [
            item
            for item in await asyncio.gather(*results_future_list)
            if item is not None
        ]
    else:
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        return status_tracker


# functions


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script


if __name__ == "__main__":
    openai_logger = logging.getLogger("openai")
    # Set the logging level for the logger to WARNING
    openai_logger.setLevel(logging.WARNING)
    with open(
        "//Users/jph/dev2/german_finetuning_data/openchat/sharegpt_gpt4.json", "r"
    ) as f:
        sharegpt_gpt4_train = json.load(f)
    # run script
    """status_tracker = asyncio.run(
        process_api_requests_from_list(
            input_list=sharegpt_gpt4_train[:20], #[:1500],
            save_filepath="test_output.jsonl",
            max_attempts=1
        )
    )"""

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

    """def postprocess_response(response: OpenAIObject, metadata) -> Any:
        #customize for results
        try:
            lang_class=LanguageClassification.from_response(response)
        except AttributeError as e:
            #logging.warning(f"Could not classify languages for {self.task_id}, parsing error")
            raise e
        res_dict={'num_languages':len(lang_class.language_codes),
                  'main_language':lang_class.main_language_code.lc,}
        #TODO rest der metadaten (token etc.) in Klasse erledigen und loggen
        #TODO Tests schreiben
        return res_dict"""

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

    def get_finished_tasks_from_file_new(file_path: str) -> set[int]:
        """Get a list of task IDs that have already been completed."""
        try:
            with open(file_path) as file:
                # `requests` will provide requests one at a time
                finished = file.__iter__()
                finished_ids = set(json.loads(line)[1]["id"] for line in finished)
                logging.debug(f"Finished Tasks found and loaded")
                return finished_ids
        except FileNotFoundError:
            logging.debug(f"No finished tasks found")
            return set()

    results = asyncio.run(
        process_api_requests_from_list(
            input_list=sharegpt_gpt4_train[:2],  # [:1500],
            max_attempts=1,
            system_msg="You are a world-class linguist and fluent in all major languages. Your job is to determine which languages are present in the user text and which one is the main language.",
            postprocess_function=postprocess_response,
            functions=[LanguageClassification.openai_schema],
            function_call={"name": "LanguageClassification"},
            save_filepath="test_output.jsonl",
            check_finished_ids=False,
            # id_field_getter=None,
            # read_finished_tasks_func=get_finished_tasks_from_file_new,
        )
    )
    """print(sharegpt_gpt4_train[0])
    print(len(results))
    print(results[0])
    print()
    print(results[1])"""
