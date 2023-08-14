# OpenAI Request Runner

A Python package designed to facilitate parallel processing of OpenAI API requests. This implementation is inspired by the [OpenAI cookbook example](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py) but offers advanced customization capabilities and integration with OpenAI Functions (leaning on the great [openai_function_call library](https://github.com/jxnl/openai_function_call)). It ensures efficient and organized interactions with the OpenAI models.
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

See `examples/classify_languages.py` and `examples/translate.py` for detailed examples of how to use the package.


The package allows for extensive customization. You can set your desired preprocessing function, postprocessing function, and other parameters to suit your specific needs.

Refer to the inline documentation and docstrings in the code for detailed information on each function and its parameters.

## Contributing

Contributions are welcome! Please open an issue if you encounter any problems or would like to suggest enhancements. Pull requests are also appreciated.

## License

MIT