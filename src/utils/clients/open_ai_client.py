from typing import Generator, List
import uuid

from utils.tools import tools_define
from utils.tools.tools_helper import extract_tool_calls_and_reupdate_output
from utils.stream_helper import process_stream_content

_open_ai_client = None


def is_loaded() -> bool:
    """Check if the OpenAI client is loaded."""
    return _open_ai_client is not None


def load_open_ai_client():
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai is not installed. Please install it using 'pip install openai'."
        )

    global _open_ai_client
    _open_ai_client = openai.OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="none",
    )


def generate(messages: List[dict], has_tool_call: bool = True) -> dict:
    response = _open_ai_client.chat.completions.create(
        messages=messages,  # type: ignore
        model="my-model",
        tools=tools_define.tools if has_tool_call else None,  # type: ignore
    ).model_dump()

    cleaned_output, tool_calls = extract_tool_calls_and_reupdate_output(
        response.get("choices", [])[0].get("message", {}).get("content", "")
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": cleaned_output,
                    "tool_calls": tool_calls,
                },
            }
        ],
    }


def generate_stream(
    messages: List[dict], has_tool_call: bool = True
) -> Generator[dict, None, None]:
    response = _open_ai_client.chat.completions.create(
        messages=messages,  # type: ignore
        model="my-model",
        tools=tools_define.tools if has_tool_call else None,  # type: ignore
        stream=True,
    )

    # Create a generator that yields content from the response
    def content_generator():
        for chunk in response:
            chunk_dict = chunk.model_dump()
            content = (
                chunk_dict.get("choices", [])[0].get("delta", {}).get("content", "")
            )
            if content:
                yield content

    # Use the common stream processor
    yield from process_stream_content(content_generator())
