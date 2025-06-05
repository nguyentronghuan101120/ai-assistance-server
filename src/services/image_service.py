import os
import time
from constants.config import IMAGE_GENERATION_CONFIG, OUTPUT_DIR
from utils.clients import image_pipeline_client


def generate_image_url(prompt: str) -> str:
    """
    Creates an image based on the specified prompt using DiffusionPipeline
    :param prompt: The prompt used for generate the image (must be in English)
    :output: URL of the new image
    """
    if not image_pipeline_client.pipeline:
        raise RuntimeError("Image pipeline not loaded")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        image = image_pipeline_client.pipeline(
            prompt=prompt,
            negative_prompt=IMAGE_GENERATION_CONFIG["negative_promt"],
            width=IMAGE_GENERATION_CONFIG["width"],
            height=IMAGE_GENERATION_CONFIG["height"],
            guidance_scale=IMAGE_GENERATION_CONFIG["guidance_scale"],
            num_inference_steps=IMAGE_GENERATION_CONFIG["num_inference_steps"],
        ).images[0]

        file_name = f"image_{int(time.time())}.png"
        image_path = os.path.join(OUTPUT_DIR, file_name)
        image.save(image_path)

        return f"{IMAGE_GENERATION_CONFIG['base_url']}{OUTPUT_DIR}/{file_name}"
    except Exception as e:
        raise RuntimeError(f"Failed to generate image: {e}")
