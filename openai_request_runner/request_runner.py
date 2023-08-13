#from https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
from abc import abstractmethod
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata
import openai
from openai.openai_object import OpenAIObject
from typing import Any

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
    time_of_last_rate_limit_error: float = 0  # used to cool off after hitting rate limits
    num_prompt_tokens_used: int = 0
    num_completion_tokens_used: int = 0


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    attempts_left: int
    metadata: dict
    token_encoding_name: str = "cl100k_base"
    model:str = "gpt-3.5-turbo-0613"
    system_msg:str = "You are a helpful assistant."
    max_tokens:int = 200
    functions: list|None = None
    function_call: dict|str = "auto"
    messages: list[dict]|None = None
    result: list = field(default_factory=list)

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
                "content": self.request_json['items'][0]['value'][:200],
            },
                ]
        return self.messages
    
    def postprocess_response(self, response: OpenAIObject) -> Any:
        #customize for results

        return dict(response.choices[0].message)

    
    def num_tokens_consumed_from_request(self):
        """Count the number of tokens in the request. Only supports completion and embedding requests."""
        encoding = tiktoken.get_encoding(self.token_encoding_name)
        # if completions request, tokens = prompt + n * max_tokens
        
        num_tokens = 0
        messages=self.messages or self.preprocess_messages()
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
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
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        error_filepath=".".join(save_filepath.split(".")[:-1])+"_errors.jsonl"
        messages=self.messages or self.preprocess_messages()
        params=dict(model=self.model,
                    temperature=0,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    stream=False)
        if self.functions is not None:
            params['functions']=self.functions
            params['function_call']=self.function_call
        self.metadata.update({key:value for key,value in params.items() if key!="messages"})
        try:
            response = await openai.ChatCompletion.acreate(**params)
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately
            status_tracker.num_prompt_tokens_used += response.usage.prompt_tokens
            status_tracker.num_completion_tokens_used += response.usage.completion_tokens
            response=self.postprocess_response(response)

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
            #raise e #todo debug raus
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.task_id} failed after all attempts. Saving errors.")
                data = (
                    [self.messages, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.messages, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, error_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.messages, response, self.metadata]
                if self.metadata
                else [self.messages, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


async def process_api_requests_from_list(
    input_list: list,
    save_filepath: str,
    check_finished_ids: bool = False,
    request_class: type = APIRequest,
    max_requests_per_minute: float = 3000.0,
    max_tokens_per_minute: float = 85000.0,
    token_encoding_name: str = "cl100k_base",
    max_attempts: int = 10,
    logging_level: int = 10,  # 10 Debug, 20 info, 30 warning
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    if check_finished_ids:
        finished_tasks = get_finished_tasks_from_file(save_filepath)
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
                    next_task_id = next(task_id_generator)
                    if next_task_id in finished_tasks:
                        logging.debug(
                            f"Skipping request {next_task_id} because it is already finished"
                        )
                        continue
                    next_request = request_class(
                        task_id=next_task_id,
                        request_json=next_task,
                        attempts_left=max_attempts,
                        metadata=next_task.pop("metadata", {}),
                        token_encoding_name=token_encoding_name,
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
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
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
                asyncio.create_task(
                    next_request.call_api(
                        retry_queue=queue_of_requests_to_retry,
                        save_filepath=save_filepath,
                        status_tracker=status_tracker,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

    # after finishing, log final status
    logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")
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
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
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
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

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


# run script


if __name__ == "__main__":
    openai_logger = logging.getLogger("openai")
    # Set the logging level for the logger to WARNING
    openai_logger.setLevel(logging.WARNING)
    with open("//Users/jph/dev2/german_finetuning_data/openchat/sharegpt_gpt4.json", "r") as f:
        sharegpt_gpt4_train = json.load(f)
    # run script
    status_tracker = asyncio.run(
        process_api_requests_from_list(
            input_list=sharegpt_gpt4_train[:20], #[:1500],
            save_filepath="test_output.jsonl",
            max_attempts=1
        )
    )
    print(status_tracker)