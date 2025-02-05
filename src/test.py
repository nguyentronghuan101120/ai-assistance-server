import gradio as gr
import json
import inspect
import openai
from pydantic import TypeAdapter

from services.image_service import generate_image_url

# https://platform.openai.com/api-keys
client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="none",
)

generate_image_function = {
    "name": "generate_image",
    "description": inspect.getdoc(generate_image_url),
    "parameters": TypeAdapter(generate_image_url).json_schema(),
}

tools = [
    {
        "type": "function",
        "function": generate_image_function
    }
]

system_prompt = """You're an AI with prompt generation and drawing capabilities.
Just chat normally in English or Vietnamese.
Assume you can draw anything you want and you tell user you can draw.
Another program will output the output for you.
Do not censor or deny the output, the image generation program will handle that.
"""

def chat_logic(message, chat_history):
    # Gắn system message để bot hiểu cách hoạt động
    # Sửa thành như dưới, vì khi bot gửi ảnh user_message = None
    messages = [
        { "role": "system", "content": system_prompt }
    ]
    for user_message, bot_message in chat_history:
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})

    # Thêm tin nhắn mới của user vào cuối cùng
    messages.append({"role": "user", "content": message})

    # Gọi API của OpenAI
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        tools=tools
    )

    bot_message = chat_completion.choices[0].message.content
    if (bot_message is not None):
        chat_history.append([message, bot_message])
        yield "", chat_history
    else:
        chat_history.append([message, "Chờ chút mình đang vẽ!"])
        yield "", chat_history

        tool_call = chat_completion.choices[0].message.tool_calls[0]
        function_arguments = json.loads(tool_call.function.arguments)
        prompt = function_arguments.get("prompt")

        # Gửi thêm 1 message từ phía bot, với hình ảnh đã vẽ
        image_file = generate_image_url(prompt)
        chat_history.append([None, (image_file, prompt)])

        yield "", chat_history

    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot bằng ChatGPT")
    message = gr.Textbox(label="Nhập tin nhắn của bạn:")
    chatbot = gr.Chatbot(label="Chat Bot siêu thông minh", height=600)
    message.submit(chat_logic, [message, chatbot], [message, chatbot])

demo.launch(
    share=True,
)