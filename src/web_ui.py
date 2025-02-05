import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# Leo Chatbot")
    message = gr.Textbox(label="Input your message:")
    chatbot = gr.Chatbot(label="Bot", height=600)
    message.submit(None, [message, chatbot], [message, chatbot])

demo.launch(
    share=True,
)