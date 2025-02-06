from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str
    is_stream: bool
