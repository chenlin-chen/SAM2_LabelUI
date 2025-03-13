import os
import shutil
import io
import json
import datetime
import gradio as gr


class UI:
    def __init__(self,
                 server_name='0.0.0.0',
                 port=7890, **kwargs):


        self.server_name = server_name
        self.port = port


    def activate_ui(self):
        demo = self.main()
        demo.queue()
        demo.launch(server_name=self.server_name, server_port=self.port)


    def main(self):
        with gr.Blocks() as demo:
            gr.Markdown(f"")

        return demo