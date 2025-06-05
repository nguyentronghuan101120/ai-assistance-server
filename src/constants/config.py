import os
from dotenv import load_dotenv
import torch

IS_APPLE_SILICON = torch.backends.mps.is_available()
IS_CUDA_AVAILABLE = torch.cuda.is_available()

TORCH_DEVICE = "cuda" if IS_CUDA_AVAILABLE else "mps" if IS_APPLE_SILICON else "cpu"
# Enable quantization for CPU/MPS
USE_QUANT = IS_CUDA_AVAILABLE

# Model optimization settings
MODEL_OPTIMIZATION = {
    "use_cache": True,
    "low_cpu_mem_usage": True,
    "torch_dtype": torch.float16 if IS_APPLE_SILICON else torch.float32,
}

IMAGE_MODEL_ID_OR_LINK = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LLM_MODEL_NAME = "NousResearch/Hermes-3-Llama-3.1-8B"
CACHE_DIR = "/tmp/cache"
VECTOR_STORE_DIR = "/tmp/vector_store"
# EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
UPLOAD_DIR = "/tmp/uploads"
OUTPUT_DIR = "/tmp/outputs"
GGUF_REPO_ID = "NousResearch/Hermes-3-Llama-3.1-8B-GGUF"
GGUF_FILE_NAME = "Hermes-3-Llama-3.1-8B.Q8_0.gguf"

load_dotenv('.env.local')  # Specify the correct file name

JINA_API_KEY = os.getenv("jina_api_key")
BRAVE_SEARCH_API_KEY = os.getenv("brave_search_api_key")

IMAGE_GENERATION_CONFIG = {
    "negative_promt": "blurry, distorted, pixelated, incomplete, poorly drawn, misaligned, weird proportions, bad perspective, unnatural colors, noisy, out of focus, glitchy, unsharp, overexposed, underexposed, poorly lit, bad composition, excessive noise, oversaturated, too dark, too bright, inconsistent lighting, discolored, overly stylized, unrealistic, awkward pose, unbalanced, mismatched, distorted features, flat, unnatural texture, chaotic, unreadable, incoherent, asymmetrical, low quality, lowres, wrong anatomy, bad anatomy, deformed, disfigured, ugly",
    "width": 512,
    "height": 512,
    "guidance_scale": 7.5,
    "num_inference_steps": 30,
    "base_url": "https://leonguyen101120-ai-assistance.hf.space:7860"
}