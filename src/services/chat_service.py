from fastapi.responses import StreamingResponse
from models.requests.chat_request import ChatRequest
from utils import client
from typing import Union

async def chat_generate(request: ChatRequest) -> Union[StreamingResponse, str]:
    """
    Generate a chat response based on the given prompt.

    If streaming is enabled, returns a StreamingResponse for live updates.
    Otherwise, returns the complete response as a string.

    Args:
        request (ChatRequest): The chat request containing the prompt and streaming flag.

    Returns:
        Union[StreamingResponse, str]: The generated chat response.
    """
    if request.is_stream:
        async def event_generator():
            # Stream the chat response in chunks
            async for chunk in client.client.chat.completions.create(
                messages=[{"role": "user", "content": request.prompt}],
                model='',
                stream=True,
            ):
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        return StreamingResponse(event_generator(), media_type='text/event-stream')
    else:
        # Non-streaming: await the complete response
        completion = await client.client.chat.completions.create(
            messages=[{"role": "user", "content": request.prompt}],
            model='',
            stream=False,
        )
        return completion.choices[0].message.content

async def get_model_info() -> dict:
    """
    Get the model information.

    Returns:
        dict: The model information.
    """
    return {
        "model_name": "GPT-2",
        "model_version": "1.0.0"
    }
