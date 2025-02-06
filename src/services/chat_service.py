from fastapi.responses import StreamingResponse
from models.requests.chat_request import ChatRequest
from utils import client

async def chat_generate(request: ChatRequest) -> str:
    """
    Stream a chat response based on the given prompt.

    Args:
        request (ChatRequest): The chat request containing the prompt.

    Returns:
        StreamingResponse: A stream of the generated chat response.
    """
    # Create async generator for streaming data
    async def event_generator():
        # Create async completion
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": request.prompt}],
            model="gpt-3.5-turbo",  # Specify a valid model
            stream=True,  # Enable streaming
        )

        async for chunk in completion:  # Process each streamed chunk
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    # Return StreamingResponse with generator and appropriate media type
    return StreamingResponse(event_generator(), media_type='text/event-stream')

async def get_model_info() -> dict:
    """
    Get the model information.

    Returns:
        dict: The model information.
    """
    return  {
        "model_name": "GPT-2",
        "model_version": "1.0.0"
    }
