import inspect

from pydantic import TypeAdapter

from services.image_service import generate_image_url

class FunctionTool:
    def __init__(self, function, name=None):
        self.function = function
        self.name = name or function.__name__
        self.description = inspect.getdoc(function)
        self.parameters = TypeAdapter(function).json_schema()

    def to_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

generate_image_tool = FunctionTool(generate_image_url)

# tools = [
#     generate_image_tool.to_dict()
# ]

tools = [
    {
        "type": "function",
        "function":{
    "name": "generate_image",
    "description": "Creates an image based on the specified prompt using DiffusionPipeline",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The prompt used for generate the image (must be in English)",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }
},
        "strict": True
    }
]
