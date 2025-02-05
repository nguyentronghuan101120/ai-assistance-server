from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.requests.chat_request import ChatRequest
from models.responses.base_response import BaseResponse
from services import chat_service
from utils import chat_client

router = APIRouter()

@router.post("/chat/", response_model=None)
async def chat(request: ChatRequest):
    data = await chat_service.chat_generate(request=request)
    yield data
    
@router.post("/chat/stream", response_model=None)  # Không sử dụng response_model
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response based on the given prompt.

    Args:
        request (ChatRequest): The chat request containing the prompt.

    Returns:
        StreamingResponse: A stream of the generated chat response.
    """
    return await chat_service.chat_generate(request=request)
    
@router.get("/get-model-info/")
async def get_model_info():
    data = await chat_service.get_model_info()
    return BaseResponse(
        data=data,
    )