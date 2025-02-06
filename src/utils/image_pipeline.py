import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

# Device setup
_device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for performance on CUDA
_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"


# Load the pipeline lazily
_pipeline = None

def get_pipeline() -> AutoPipelineForText2Image:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = AutoPipelineForText2Image.from_pretrained(
                _model_id,
                torch_dtype=torch.bfloat16,
                # variant="fp16",
                # safety_checker=True,
                use_safetensors=True
            )
            _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(_pipeline.scheduler.config)
            # _pipeline.enable_model_cpu_offload()
            _pipeline.to(_device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")
    return _pipeline

pipeline = get_pipeline()
