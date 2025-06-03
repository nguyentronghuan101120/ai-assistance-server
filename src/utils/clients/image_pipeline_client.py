from constants.config import IMAGE_MODEL_ID_OR_LINK
from utils.timing import measure_time


pipeline = None


def load_pipeline():
    global pipeline
    try:
        import torch
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
            StableDiffusionPipeline,
        )

        torch.backends.cuda.matmul.allow_tf32 = (
            True  # Enable TF32 for performance on CUDA
        )
    except ImportError:
        raise ImportError(
            "diffusers is not installed. Please install it using 'pip install diffusers'."
        )

    with measure_time("Load image pipeline"):
        pipeline = StableDiffusionPipeline.from_pretrained(
            IMAGE_MODEL_ID_OR_LINK,
            torch_dtype=torch.bfloat16,
            variant="fp16",
            # safety_checker=True,
            use_safetensors=True,
        )


def clear_resources():
    global pipeline
    pipeline = None
