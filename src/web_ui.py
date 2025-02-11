import gradio as gr
import json

from models.requests.chat_request import ChatRequest
from services import chat_service, image_service
from utils import client
from utils.prompts import system_prompt


def chat_logic(message, chat_history):
    # Build the conversation messages with the system prompt for context.
    messages = [{"role": "system", "content": system_prompt}]
    for user_message, bot_message in chat_history:
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})
    # Append the new user message.
    messages.append({"role": "user", "content": message})

    # Call the OpenAI API.
    bot_message =  chat_service.chat_generate(
        request=ChatRequest(prompt=messages, is_stream=False)
    )

    # If no completion is returned, raise an exception.
    if (bot_message is not None):
        chat_history.append([message, bot_message])
        yield "", chat_history

    # Check if the response has tool_calls (for drawing) or a text response.
    else:
        chat_history.append([message, "Please wait while I'm drawing..."])
        yield "", chat_history

        tool_call = bot_message.tool_calls[0]
        function_arguments = json.loads(tool_call.function.arguments)
        prompt = function_arguments.get("prompt")

        # Gửi thêm 1 message từ phía bot, với hình ảnh đã vẽ
        image_file = image_service.generate_image_url(prompt)
        chat_history.append([None, (image_file, prompt)])

        yield "", chat_history

    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# Chatbot bằng ChatGPT")
    message = gr.Textbox(label="Nhập tin nhắn của bạn:")
    chatbot = gr.Chatbot(label="Chat Bot siêu thông minh", height=600)
    message.submit(chat_logic, [message, chatbot], [message, chatbot])

demo.launch(share=True)
