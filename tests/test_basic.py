from openai_request_runner import process_api_requests_from_list
import asyncio

#Needs OpenAI API Key in environment variable OPENAI_API_KEY
def test_basic_functionality():
    example_input = [{"id": 0, "prompt": "What is 1+1?"}]
    results = asyncio.run(
        process_api_requests_from_list(
            example_input
        )
    )
    assert("2" in results[0][1]["content"]) #type: ignore
