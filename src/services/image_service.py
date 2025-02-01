import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from PIL.Image import Image

from models.requests.image_request import ImageRequest
# from diffusers import BitsAndBytesConfig, SD3Transformer2DModel

# Device setup
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for performance on CUDA
model_id = "stabilityai/stable-diffusion-xl-base-0.9"  # Update to actual path if local


# Load the pipeline lazily
_pipeline = None

def get_pipeline() -> AutoPipelineForText2Image:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                variant="fp16",
                safety_checker=True,
                use_safetensors=True
            )
            _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(_pipeline.scheduler.config)
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
            # generator=torch.Generator(device=device).manual_seed(-1)
        ).images[0]
    except Exception as e:
        raise RuntimeError(f"Failed to generate image: {e}")
