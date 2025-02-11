import inspect

from pydantic import TypeAdapter

from services.image_service import generate_image_url


generate_image_function = {
    "name": "generate_image",
    "description": inspect.getdoc(generate_image_url),
    "parameters": TypeAdapter(generate_image_url).json_schema(),
}

tools = [
    {
        "type": "function",
        "function": generate_image_function
    }
]