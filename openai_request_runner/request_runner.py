# inspired by https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
import asyncio
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)
from typing import Any, Callable, Iterable, Optional, Union

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from openai_request_runner.utils import append_to_jsonl

aclient = AsyncOpenAI()

DEFAULT_RATE_LIMITS = {
    "gpt-3.5": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 1000000,
    },
    "gpt-4": {"max_requests_per_minute": 10000, "max_tokens_per_minute": 300000},
}

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

    task_id: Union[int, str]
    request_json: dict
    attempts_left: int
    metadata: dict
    preprocess_messages: Callable[[dict, dict], list[dict]]
    postprocess_response: Callable[[ChatCompletion, dict, dict], dict]
    messages: Optional[list[dict]] = None
    result: list = field(default_factory=list)
    return_results: bool = False
    debug: bool = False

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
        save_filepath: Optional[str],
        raw_request_filepath: Optional[str],
        error_filepath: str,
        status_tracker: StatusTracker,
        temperature: float = 0,
    ):
        """Calls the OpenAI API and saves results."""
        logging.debug(f"Starting request #{self.task_id}")
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
            temperature=temperature,
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
            if self.debug:
                logging.debug(f"Request params: {params}")
            response = await aclient.chat.completions.create(**params)
            assert isinstance(response, ChatCompletion)
            status_tracker.num_prompt_tokens_used += response.usage.prompt_tokens
            status_tracker.num_completion_tokens_used += (
                response.usage.completion_tokens
            )
            if raw_request_filepath is not None:
                tmp_raw_request = response.model_dump()
                tmp_raw_request["request"] = params
                tmp_raw_request["task_id"] = self.task_id
                append_to_jsonl(tmp_raw_request, raw_request_filepath)
        except Exception as e:  # handling request errors
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
            response = None
            if self.debug:
                logging.warning(params)
                raise e

        if response is not None:
            if self.debug:
                logging.debug(f"Response before postprocessing: {response}")
            try:
                response = self.postprocess_response(
                    response, self.request_json, self.metadata
                )
            except Exception as e:
                # handling postporcessing errors
                logging.warning(
                    f"Error during postprocessing for request {self.task_id}: {e}"
                )
                status_tracker.num_other_errors += 1
                error = e
                response = None
                if self.debug:
                    raise e

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
            return None
        else:
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

            if not save_filepath:
                logging.info(f"Request {self.task_id} suceeded.")
            else:
                logging.info(f"Request {self.task_id} saved to {save_filepath}")
                append_to_jsonl(response, save_filepath)  # type: ignore
            return response


def preprocess_messages(request_json: dict, metadata: dict) -> list[dict]:
    # preprocess input to openai format
    messages = [
        {
            "role": "system",
            "content": metadata["system_msg"],
        },
        {
            "role": "user",
            "content": request_json["prompt"],
        },
    ]
    return messages


def postprocess_response_default(
    response: ChatCompletion, request_json: dict, metadata: dict
) -> dict:
    # customize for results
    tmperg = dict(response.choices[0].message)
    tmperg["task_id"] = metadata["task_id"]
    return tmperg


def get_finished_tasks_from_file(file_path: str) -> set[int]:
    """Get a list of task IDs that have already been completed."""
    try:
        with open(file_path) as file:
            # `requests` will provide requests one at a time
            finished = file.__iter__()
            finished_ids = set(
                int(list(json.loads(line).keys())[0]) for line in finished
            )
            logging.info("Finished Tasks found and loaded")
            return finished_ids
    except FileNotFoundError:
        logging.info("No finished tasks found")
        return set()


def get_id_from_finished_default(result: dict) -> Union[int, str]:
    return result["task_id"]


async def process_api_requests_from_list(
    inputs: Iterable[dict],
    save_filepath: Optional[str] = None,
    raw_request_filepath: Optional[str] = None,
    error_filepath: str = "tmp_error_log.jsonl",
    token_encoding_name: str = "cl100k_base",
    model: str = "gpt-3.5-turbo-1106",
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    id_field_getter: Optional[Callable[[dict], Union[str, int]]] = lambda x: x["id"],
    functions: Optional[list] = None,
    function_call: Union[dict, str] = "auto",
    preprocess_function: Callable[[dict, dict], list[dict]] = preprocess_messages,
    postprocess_function: Callable[
        [ChatCompletion, dict, dict], dict
    ] = postprocess_response_default,
    check_finished_ids: bool = False,
    get_id_from_finished: Callable[
        [dict], Union[int, str]
    ] = get_id_from_finished_default,
    finished_ids: Optional[set[int]] = None,
    max_requests_per_minute: Optional[float] = None,
    max_tokens_per_minute: Optional[float] = None,
    max_attempts: int = 2,
    logging_level: int = 20,
    num_max_requests: Optional[int] = None,
    temperature: float = 0,
    debug: bool = False,
) -> list[Any]:
    """
    Processes a list of API requests in parallel, ensuring that they stay under rate limits.
    Saves the results to a file or returns them, depending on the `save_filepath` argument.

    Args:
        inputs (Iterable[dict]): An iterable of API request payloads to be processed.

        save_filepath (str | None, optional): Path to save the processed results. If None, results will be returned. Defaults to None.

        error_filepath (str, optional): Path to save any errors encountered during processing. Defaults to "error_log.jsonl".

        token_encoding_name (str, optional): Name of the token encoding scheme to be used. Defaults to "cl100k_base".

        model (str, optional): Name of the OpenAI model to use. Defaults to "gpt-3.5-turbo-0613".

        system_prompt (str, optional): System message to be sent before any other message. Defaults to "You are a helpful assistant.".

        max_tokens (int, optional): Maximum tokens allowed in a single request. Defaults to 200.

        id_field_getter (Callable[[dict], str | int] | None, optional): Function to extract the ID from the input request. Defaults to lambda x: x["id"].

        functions (list | None, optional): List of functions to be used in the API request (if any). Defaults to None.

        function_call (dict | str, optional): Function call details to be used in the API request. Defaults to "auto".

        preprocess_function (Callable, optional): Function to preprocess the input messages before sending the request.
            Example Implementation:
                def preprocess_messages_example(request_json: dict, metadata: dict) -> list[dict]:
                    '''
                    A simplified preprocess function that takes the 'content' field from the request_json
                    and returns it wrapped in a list of dictionaries.
                    '''
                    messages = [
                        {
                            "role": "user",
                            "content": request_json["content"],
                        }
                    ]
                    return messages
            Defaults to preprocess_messages_sharegpt.

        postprocess_function (Callable, optional): Function to process the response received from the API. It can be used to extract specific parts of the response, reformat it, or add additional information.
            Example Implementation:
                def postprocess_response_example(response: ChatCompletion, request_json: dict, metadata: dict) -> dict:
                    '''
                    A simplified postprocess function that extracts the message content from the API response.
                    '''
                    return {"response_content": response.choices[0].message['content']}
            Defaults to postprocess_response_default.

        check_finished_ids (bool, optional): Whether to check for IDs of already completed tasks and skip them. Defaults to False.

        get_id_from_finished (Callable, optional): Function to extract the ID from a finished result. If you're processing a list of tasks and saving the results, you might want to skip the tasks that were already completed in a previous run. This function helps in identifying the IDs of those completed tasks.
            Example Implementation:
                def get_id_from_finished_example(result_list: list) -> int:
                    '''
                    A simplified function to extract the 'task_id' from the metadata of a result.
                    '''
                    return result_list[2]["metadata"]["task_id"]
            Defaults to get_id_from_finished_default.

        finished_ids (set[int] | None, optional): Set of IDs of already completed tasks, if available. Defaults to None.

        max_requests_per_minute (float | None, optional): Maximum number of requests allowed per minute. If None, will use the rate limits for the specified model. Defaults to None.

        max_tokens_per_minute (float | None, optional): Maximum number of tokens allowed per minute. If None, will use the rate limits for the specified model. Defaults to None.

        max_attempts (int, optional): Maximum number of attempts for each request before considering it as failed. Defaults to 2.

        logging_level (int, optional): Logging level to be used. Defaults to 10 (DEBUG level).

        num_max_requests (int | None, optional): Maximum number of requests to process. If None, will process all the inputs. Defaults to None.

    Returns:
        StatusTracker | List[dict]: If `save_filepath` is provided, returns an instance of StatusTracker indicating the processing status.
                                   Otherwise, returns a list of dictionaries containing the results.
    """
    if save_filepath is None:
        write_to_file = False
    else:
        write_to_file = True
    if max_requests_per_minute is None:
        for model_name_stub in DEFAULT_RATE_LIMITS.keys():
            if model.startswith(model_name_stub):
                max_requests_per_minute = DEFAULT_RATE_LIMITS[model_name_stub][
                    "max_requests_per_minute"
                ]
                break
        logging.info(f"Max requests per minute set to {max_requests_per_minute}")
    if max_requests_per_minute is None:
        max_requests_per_minute = 10000
        logging.warning(
            f"Max requests per minute not set and model {model} not found in defaults, using default value of 10000 rpm/1m tpm"
        )
    if max_tokens_per_minute is None:
        for model_name_stub in DEFAULT_RATE_LIMITS.keys():
            if model.startswith(model_name_stub):
                max_tokens_per_minute = DEFAULT_RATE_LIMITS[model_name_stub][
                    "max_tokens_per_minute"
                ]
                break
        logging.info(f"Max tokens per minute set to {max_tokens_per_minute}")
    if max_tokens_per_minute is None:
        max_tokens_per_minute = 10000000

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
                logging.debug("No finished tasks found")
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
    max_requests_not_reached = (
        True  # after max requests reached, we'll stop reading file
    )
    logging.debug("Initialization complete.")

    task_generator = iter(inputs)
    logging.info("Entering main loop")

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.info(f"Retrying request {next_request.task_id}")
            elif (
                file_not_finished
                and max_requests_not_reached
                and (
                    num_max_requests is None
                    or num_max_requests > status_tracker.num_tasks_started
                )
            ):
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
                    metadata["system_msg"] = system_prompt
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
                        debug=debug,
                    )
                    status_tracker.num_tasks_started += 1
                    if status_tracker.num_tasks_started % 200 == 0:
                        logging.info(status_tracker)
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.info("Request List exhausted")
                    file_not_finished = False
            elif (
                file_not_finished
                and num_max_requests is not None
                and max_requests_not_reached
            ):
                logging.info(f"Max number of requests reached ({num_max_requests})")
                max_requests_not_reached = False
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
                        raw_request_filepath=raw_request_filepath,
                        error_filepath=error_filepath,
                        status_tracker=status_tracker,
                        temperature=temperature,
                    )
                )
                results_future_list.append(task)
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # if max requests reached, break
        if num_max_requests is not None:
            if (
                status_tracker.num_tasks_succeeded + status_tracker.num_tasks_failed
            ) >= num_max_requests:
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
            logging.warning(
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
    logging.info(status_tracker)
    if write_to_file:
        logging.info(f"Parallel processing complete. Results saved to {save_filepath}")

    """result_list = []
    for coroutine in asyncio.as_completed(results_future_list):
        try:
            exception = coroutine.exception()
            print(exception)
            result = await coroutine
            if result is not None:
                result_list.append(result)
            continue
        except Exception as e:
            logging.warning(f"Error in coroutine: {e}")
            if debug:
                raise e
            continue
    return result_list"""

    return [
        item
        for item in await asyncio.gather(*results_future_list, return_exceptions=True)
        if item is not None
    ]


# functions


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


# minimal usage example
if __name__ == "__main__":
    example_input = [{"id": 0, "prompt": "What is 1+1?"}]
    results = asyncio.run(
        process_api_requests_from_list(
            example_input, system_prompt="Translate input to French"
        )
    )
    print(results[0]["content"])  # type: ignore
    # "Qu'est-ce que 1+1 ?"

    # see examples/ for advanced usage
