from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: list[dict]
    is_stream: bool
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": [{"role": "user", "content": "Hello, how are you?"}],
                    "is_stream": False
                }
            ]
        }
    }

class ChatStreamRequest(BaseModel):
    prompt: list[dict]
    is_stream: bool
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": [{"role": "user", "content": "Hello, how are you?"}],
                    "is_stream": True
                }
            ]
        }
    }
