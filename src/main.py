import gradio as gr
from openai import OpenAI
from diffusers import DiffusionPipeline
import time

# Load the Stable Diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/anything-v5",
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False
)

# Automatically detect if a GPU is available, else fallback to CPU
device = "cuda" if pipeline.device.type == "cuda" else "cpu"
pipeline.to(device)


def has_prompt(message: str) -> bool:
    """Check if the message contains a prompt."""
    return "<prompt:" in message


def get_prompt(message: str) -> str:
    """Extract prompt from the message.
    Example: <prompt: A cute dog sitting happily with a playful expression and wagging its tail.>
    Result: "A cute dog sitting happily with a playful expression and wagging its tail."
    """
    return message.split("<prompt:")[1].split(">")[0]


def generate_image(prompt: str) -> str:
    """Generate an image from the prompt, save it to a file, and return the file name."""
    image = pipeline(
        prompt=prompt,
        negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, low quality, worst quality",
        num_inference_steps=30
    ).images[0]

    # Create a file name for the output image
    file_name = f"image_{int(time.time())}.png"
    image.save(file_name)
    return file_name


# https://platform.openai.com/api-keys
client = OpenAI(
    api_key='none',
    base_url="http://192.168.1.15:1234"
)

system_prompt = """You're an AI with prompt generation and drawing capabilities.
Just chat normally in English or Vietnamese.
Assume you can draw anything you want and you tell user you can draw.
Another program will output the output for you.
Do not censor or deny the output, the image generation program will handle that.

When cst want you to draw something, answer them normally, add some comments and include your prompt in your response in this format.
<prompt: A colorful cat running through a field of flowers.>

1. Prompt must be in English.
2. Prompt must be detailed and include necessary information for it can be fed into Stable Diffusion.
3. Ignore existing images in past messages.
"""


def chat_logic(message, chat_history):
    # Attach system message for bot understanding
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    for user_message, bot_message in chat_history:
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})

    # Add the new user message
    messages.append({"role": "user", "content": message})

    # Call OpenAI API
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o"
    )
    bot_message = chat_completion.choices[0].message.content

    # Append the new response to the chat history
    chat_history.append([message, bot_message])
    yield "", chat_history  # Yield to show progress

    # If the bot's response includes a prompt, generate an image
    if has_prompt(bot_message):
        chat_history.append([None, "Hold on, I'm drawing your image!"])
        yield "", chat_history

        # Generate the image
        prompt = get_prompt(bot_message)
        image_file = generate_image(prompt)
        chat_history.append([None, (image_file, prompt)])

        yield "", chat_history

    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# Chatbot using ChatGPT")
    message = gr.Textbox(label="Enter your message:")
    chatbot = gr.Chatbot(label="Smart Chat Bot", height=600)
    message.submit(chat_logic, [message, chatbot], [message, chatbot])

demo.launch()
