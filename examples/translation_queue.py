#TODO FIX

import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata
from langchain.callbacks import get_openai_callback
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

system_template = """You are a world class translator and your job is to translate the English user input into German. 

Please adhere to the following rules:
- Do not change the meaning of the text
- Do not add or remove information
- Do not change the style of the text
- Do not change the tone of the text
- Do only translate english text to German and leave all other text as is
- Do not translate any names
- Use always the direct/casual form of German address ("Du" instead of "Sie") except in formal settings (e.g. if the user asks to write an official letter)
- Do not translate any computer code (most often denoted with triple backticks like ```) and comments in the code
- Do not translate the following tokens: "--!--", "Human:", "Assistant:", "--!!!--"
- Please answer ONLY with the translation of the user input and nothing else!
- When in doubt don't translate word for word but rather translate the meaning of the sentence in a way that sounds natural in German!

The following (between "START OF CONVERSATION" and "END OF CONVERSATION") is an excerpt of a conversation between a user and an assistant and your job is to translate all English text correctly into German according to the given rules above!
"""

user_template = """--------START OF CONVERSATION--------
{text}
--------END OF CONVERSATION--------
Please translate the complete English text above correctly into German now and remember the rules from above (use "Du" instead of "Sie" except in formal settings; don't translate code, names, special tokens or non-english text; answer only with the translation and nothing else; include everything between "START OF CONVERSATION" and "END OF CONVERSATION" including the user query in your translation)!"""

# input variables: text
chat_prompt_translation = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template),
    ]
)


async def process_api_requests_from_list(
    input_list: list,
    save_filepath: str,
    llm: ChatOpenAI = ChatOpenAI( # type: ignore
        model_name="gpt-3.5-turbo-0613",  # type: ignore
        temperature=0,
        request_timeout=120,
        max_retries=0,
    ), # type: ignore
    prompt_template: ChatPromptTemplate = chat_prompt_translation,
    max_requests_per_minute: float = 3000.0,
    max_tokens_per_minute: float = 85000.0,
    token_encoding_name: str = "cl100k_base",
    max_attempts: int = 10,
    logging_level: int = 10,  # 10 Debug, 20 info, 30 warning
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.01  # 1 ms limits max throughput to 1,000 requests per second
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # initialize trackers
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

    finished_tasks = get_finished_tasks_from_file(save_filepath)

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug("Initialization complete.")

    translation_tasks = iter(input_list)
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
                    next_translation = next(translation_tasks)
                    next_task_id = next(task_id_generator)
                    if next_task_id in finished_tasks:
                        logging.debug(
                            f"Skipping request {next_task_id} because it is already finished"
                        )
                        continue
                    next_request = TranslationRequest(
                        task_id=next_task_id,
                        text=next_translation,
                        chain=chain,
                        token_consumption=num_tokens_consumed_from_request(
                            next_translation, llm, prompt_template
                        ),
                        attempts_left=max_attempts,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Input exhausted")
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
            next_request_tokens = next_request.token_consumption
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
                    next_request.generate_translation(
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
        seconds_since_last_status_update = (
            time.time() - status_tracker.time_of_last_status_update
        )
        if seconds_since_last_status_update > 15:
            logging.info(status_tracker)
            status_tracker.time_of_last_status_update = time.time()
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
    logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )
    logging.info(
        f"Total cost: {status_tracker.total_cost:.2f} USD; {status_tracker.num_tokens_used} tokens used"
    )
    return status_tracker


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
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits
    time_of_last_status_update: float = 0  # used to print status updates periodically
    num_tokens_used: int = 0
    total_cost: float = 0.0


@dataclass
class TranslationRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    text: str
    chain: LLMChain
    token_consumption: int
    attempts_left: int
    result: str = ""

    def preprocess_text(self) -> dict[str, str]:
        try:
            assert "<|end_of_turn|>" in self.text
        except AssertionError as e:
            logging.warning(
                f"Request {self.task_id} - no end-of-turn tokens found in original text"
            )
            self.attempts_left = 0
            raise e
        text = self.text.replace("<s>", "--!--")
        text = text.replace("<|end_of_turn|>", "--!!!--")
        return {"text": text}

   def postprocess_text(self, response: dict[str, str]) -> str:
        text = response["text"]
        text = text.replace("--!--", "<s>")
        text = text.replace("--!!!--", "<|end_of_turn|>")
        try:
            assert "<s>" in text
            assert "<|end_of_turn|>" in text
        except AssertionError as e:
            logging.warning(
                f"Request {self.task_id} - no start/end-of-turn tokens found in response"
            )
            self.attempts_left = 0
            raise e 
        return text

    async def generate_translation(
        self,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls langchain and generates translation."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            translation_dict = self.preprocess_text()
            with get_openai_callback() as cb:
                response = await self.chain.acall(translation_dict)
            status_tracker.num_tokens_used -= cb.total_tokens
            status_tracker.total_cost += cb.total_cost
            self.result = self.postprocess_text(response)
        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
                logging.error(
                    f"Request {self.task_id} failed after all attempts. Last Error: {error}.\n Text start: {self.text[:100]}"
                )
        else:
            append_to_jsonl({self.task_id: self.result}, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved")


# functions


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    text: str,
    llm: ChatOpenAI,
    prompt_template: ChatPromptTemplate,
) -> int:
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    messages = prompt_template.format_messages(text=text)
    return llm.get_num_tokens_from_messages(messages)


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
    with open("/Users/jph/dev2/finetune_orcamini/openchat.train.text.json", "r") as f:
        openchat_train = json.load(f)
    # run script
    status_tracker = asyncio.run(
        process_api_requests_from_list(
            input_list=openchat_train, #[:1500],
            save_filepath="/Users/jph/dev2/finetune_orcamini/openchat.train.text_translated.jsonl",
        )
    )
    print(status_tracker)
