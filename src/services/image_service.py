import torch
from diffusers import StableDiffusion3Pipeline
from PIL.Image import Image
from models.image_models import ImageRequest
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel

# Device setup
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for performance on CUDA
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"  # Update to actual path if local


# Load the pipeline lazily
_pipeline = None

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

def get_pipeline() -> StableDiffusion3Pipeline:
    global _pipeline
    if _pipeline is None:
        try:
            dtype = torch.bfloat16
            _pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                transformer=model_nf4,
            )
            # _pipeline.enable_model_cpu_offload()
            _pipeline.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")
    return _pipeline

async def generate_image(imgRequest: ImageRequest) -> Image:
    pipeline = get_pipeline()
    try:
        # Generate image based on input request
        return pipeline(
            prompt=imgRequest.prompt,
            negative_prompt=imgRequest.negative_prompt,
            width=imgRequest.width,
            height=imgRequest.height,
            guidance_scale=imgRequest.guidance_scale,
            num_inference_steps=imgRequest.num_inference_steps,
        ).images[0]
    except Exception as e:
        raise RuntimeError(f"Failed to generate image: {e}")
