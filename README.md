# OpenAI Request Runner


[![Pypi](https://img.shields.io/pypi/v/openai-request-runner?color=g
)](https://pypi.org/project/openai-request-runner/)
[![CI](https://github.com/jphme/openai_request_runner/actions/workflows/test.yml/badge.svg)](https://github.com/jphme/openai_request_runner/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/pypi/status/openai-request-runner
)](https://pypi.org/project/openai-request-runner/)


[![Twitter](https://img.shields.io/twitter/follow/jphme
)](https://twitter.com/jphme)

A lightweight Python package designed to facilitate parallel processing of OpenAI API requests. This implementation is inspired by the [OpenAI cookbook example](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py) but offers advanced customization capabilities and integration with OpenAI Functions (leaning on the great [openai_function_call library](https://github.com/jxnl/openai_function_call)). It ensures efficient and organized interactions with the OpenAI models.
Features

* Parallel Processing: Handle multiple OpenAI API requests concurrently.
* Rate Limiting: Adheres to rate limits set by the OpenAI API.
* Advanced Customization: Allows for detailed input preprocessing and API response postprocessing.
* OpenAI Functions: Seamlessly integrates with OpenAI Functions for added capabilities.
* Error Handling: Efficiently manage and log errors, including rate limit errors.
* Extendable: Easily integrate with custom schemas and other extensions.

## Installation
### Using pip (wip)

```bash
pip install openai_request_runner
```

### Git

```bash
pip install git@https://github.com/jphme/openai_request_runner
```
### Using poetry

For local development and testing:

```bash
poetry install
```
## Usage

Minimal example:
```python
import asyncio
from openai_request_runner import process_api_requests_from_list

example_input = [{"id": 0, "prompt": "What is 1+1?"}]
results = asyncio.run(
    process_api_requests_from_list(
        example_input, system_prompt="Translate input to French"
    )
)
#or in a notebook:
#results = await process_api_requests_from_list(...

print(results[0]["content"])
# "Qu'est-ce que 1+1 ?"
```

See `examples/classify_languages.py` and `examples/translate.py` for detailed examples of how to use the package for advanced usecases.

The package allows for extensive customization. You can set your desired preprocessing function, postprocessing function, and other parameters to suit your specific needs.

Refer to the inline documentation and docstrings in the code for detailed information on each function and its parameters.

### Run inside a notbook

If you want to run openai_request_runner inside a notebook, use `nest_asyncio` like this:

```python
import nest_asyncio
nest_asyncio.apply()
```


### Run Tests

```bash
poetry run pytest tests/
``````

## Contributing

Contributions are welcome! Please open an issue if you encounter any problems or would like to suggest enhancements. Pull requests are also appreciated.

## License

MIT