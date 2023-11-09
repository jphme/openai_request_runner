import asyncio
import unittest

from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import Field

from openai_request_runner import process_api_requests_from_list
from openai_request_runner.openaischema import OpenAISchema


# Needs OpenAI API Key in environment variable OPENAI_API_KEY
class TestOpenAIRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.example_input = [{"id": 0, "prompt": "What is 1+1?"}]
        return super().setUp()

    def test_basic_functionality(self):
        results = asyncio.run(process_api_requests_from_list(self.example_input))
        assert "2" in results[0]["content"]  # type: ignore

    def test_get_responses_function(self):
        class Answer(OpenAISchema):
            """Answer for the user query."""

            answer: str = Field(
                ...,
                description="The answer for the user query.",
            )

        try:
            results = asyncio.run(
                process_api_requests_from_list(
                    self.example_input,
                    functions=[Answer.openai_schema],
                    function_call={"name": Answer.openai_schema["name"]},
                    max_tokens=10,
                )
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(responses, [{"answer": "Okay."}, {"answer": "Yes."}])
        except (APITimeoutError, APIConnectionError, APIStatusError) as e:
            self.skipTest(f"Skipped due to Connection Error: {e}")


if __name__ == "__main__":
    unittest.main()
