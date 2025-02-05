import time
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
model_id = "stablediffusionapi/anything-v5"  # Update to actual path if local


# Load the pipeline lazily
_pipeline = None

def get_pipeline() -> AutoPipelineForText2Image:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                # variant="fp16",
                # safety_checker=True,
                use_safetensors=True
            )
            _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(_pipeline.scheduler.config)
            # _pipeline.enable_model_cpu_offload()
            _pipeline.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")
    return _pipeline

pipeline = get_pipeline()

async def generate_image(imgRequest: ImageRequest) -> Image:

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
    
def generate_image_url(prompt: str) -> str:
    """
    Creates an image based on the specified prompt using DiffusionPipeline
    :param prompt: The prompt used for generate the image (must be in English)
    :output: URL of the new image
    """
    image = pipeline(
            prompt=prompt,
            negative_prompt='blurry, distorted, pixelated, incomplete, poorly drawn, misaligned, weird proportions, bad perspective, unnatural colors, noisy, out of focus, glitchy, unsharp, overexposed, underexposed, poorly lit, bad composition, excessive noise, oversaturated, too dark, too bright, inconsistent lighting, discolored, overly stylized, unrealistic, awkward pose, unbalanced, mismatched, distorted features, flat, unnatural texture, chaotic, unreadable, incoherent, asymmetrical, low quality, lowres, wrong anatomy, bad anatomy, deformed, disfigured, ugly',
            width=512,
            height=512,
            guidance_scale=7.5,
            num_inference_steps=30,
            # generator=torch.Generator(device=device).manual_seed(-1)
        ).images[0]

    file_name = f"image_{int(time.time())}.png"
    image.save(file_name)
    return file_name
