import base64
import io
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models.requests.image_request import ImageRequest
from services import image_service

router = APIRouter()

@router.post("/generate/")
async def generate_image(imgRequest: ImageRequest):
    image = await image_service.generate_image(imgRequest=imgRequest)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")

@router.post("/generatebase64/")
async def generate_base64_image(imgRequest: ImageRequest):
    image = await image_service.generate_image(imgRequest=imgRequest)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return {"image": img_str}