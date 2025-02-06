from fastapi import APIRouter
from models.requests.chat_request import ChatRequest
from models.responses.base_response import BaseResponse
from services import chat_service

router = APIRouter()

@router.post("/chat/", response_model=None)
async def chat(request: ChatRequest):
    """
    Generate a chat response for the given prompt.
    This endpoint yields the response data directly.

    Args:
        request (ChatRequest): The chat request containing the prompt.

    Yields:
        The generated chat response data.
    """
    data = await chat_service.chat_generate(request=request)
    yield data
    
@router.post("/chat/stream", response_model=None)  
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response based on the given prompt.
    This endpoint returns a StreamingResponse for real-time chat generation.

    Args:
        request (ChatRequest): The chat request containing the prompt.

    Returns:
        StreamingResponse: A stream of the generated chat response.
    """
    return await chat_service.chat_generate(request=request)
    
@router.get("/get-model-info/")
async def get_model_info():
    """
    Get information about the currently loaded chat model.
    Returns details like model name and version.

    Returns:
        BaseResponse: A response containing the model information in the data field.
    """
    data = await chat_service.get_model_info()
    return BaseResponse(
        data=data,
    )