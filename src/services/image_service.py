import torch
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from PIL import Image as PILImage
from models.image_models import ImageRequest

# Set up device selection
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load the model pipeline only once and transfer to the selected device
pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,  # Use float16 for GPU, float32 for CPU
    use_safetensors=True,
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.to(device)

# Enable mixed-precision (autocast) for more efficient computation on supported hardware
torch.backends.cudnn.benchmark = True  # Use the optimal algorithm for your hardware

async def generate_image(imgRequest: ImageRequest) -> PILImage:
    # Use mixed precision to speed up computations if available
    with torch.autocast(device_type=device, dtype=torch.float16 if device != 'cpu' else torch.float32):
        # Run the inference and generate the image
        image = pipeline(
            prompt=imgRequest.prompt,
            negative_prompt=imgRequest.negative_prompt,
            width=imgRequest.width,
            height=imgRequest.height,
            guidance_scale=imgRequest.guidance_scale,
            num_inference_steps=imgRequest.num_inference_steps,
        ).images[0]
    return image
