import gradio as gr
from rag_engine_app import get_answer

with gr.Blocks(theme=gr.themes.Base(primary_hue="red"),
    css="""
        #chat-container {
            max-width: 60%;
            margin-left: auto;
            margin-right: auto;
        }
        """
    ) as demo:
    with gr.Column(elem_id="chat-container"):
        gr.Markdown(
            """
            <div style="font-size: 30px;">
            <strong>🎒 Welcome to the Marbet RAG Assistant!</strong><br>
            Ask anything about the trips — activities, schedules, visa steps. 
            I’ll give you accurate answers based on the official documents.
            </div>
            """,
            elem_id="description"
        )
        gr.ChatInterface(
            fn=get_answer,
            type='messages',
            chatbot=gr.Chatbot(height=500),
            textbox=gr.Textbox(
                placeholder='Ask anything',
                container=True,
                scale=1,
                lines=1
            )
        )

demo.launch()
