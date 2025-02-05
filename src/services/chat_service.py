from fastapi.responses import StreamingResponse
from models.requests.chat_request import ChatRequest
from utils import chat_client

async def chat_generate(request: ChatRequest) -> str:
    """
    Stream a chat response based on the given prompt.

    Args:
        request (ChatRequest): The chat request containing the prompt.

    Returns:
        StreamingResponse: A stream of the generated chat response.
    """
    # Tạo async generator để gửi dữ liệu từng phần
    async def event_generator():
        async for chunk in chat_client.generate_chat_stream(request.prompt):
            yield chunk  # Gửi từng phần của dữ liệu

    # Trả về StreamingResponse với generator và media_type phù hợp
    return StreamingResponse(event_generator(), media_type='text/event-stream')

async def get_model_info() -> dict:
    """
    Get the model information.

    Returns:
        dict: The model information.
    """
    
    return chat_client.get_model_info()
