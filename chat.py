"""
Adapted from: https://github.com/DAMO-NLP-SG/Video-LLaMA/blob/main/demo_video.py
"""
import argparse
import os

import numpy as np
import torch
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation
import decord
decord.bridge.set_bridge('torch')

#%%
from video_llama.models import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=False, default="eval_configs/askvideos_instruct_ft.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False), gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_video(gr_video, text_input, chat_state,chatbot):
    chat_state = default_conversation.copy()        
    if gr_video is not None:
        print(gr_video)
        chatbot = chatbot + [((gr_video,), None)]
        #chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        chat_state.system =  "You are able to understand the visual content that the user provides. Answer the questions from the video."
        img_list = []
        llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list, num_frames=16)
        return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    return None, None, gr.update(interactive=True), chat_state, None

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, temperature):
    print('Chat state: ', chat_state)
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=1,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    print(chat_state.get_prompt())
    print(chat_state)
    return chatbot, chat_state, img_list

title = """

<h1 align="center">AskVideos Instruct demo</h1>

<h5 align="center">  Introduction: Ask-Videos Instruct is a based on the Video-LLaMA model and is finetuned with more data to work on YouTube videos.</h5> 

"""

cite_markdown = ("""
## Credits
Adapted from [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA/)
""")

with gr.Blocks() as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=0.5):
            video = gr.Video()

            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='Ask-Videos')
            text_input = gr.Textbox(label='User', placeholder='Upload your video first, or directly click the examples at the bottom of the page.', interactive=False)
            
        
    gr.Markdown(cite_markdown)
    upload_button.click(upload_video, [video, text_input, chat_state, chatbot], [video, text_input, upload_button, chat_state, img_list, chatbot])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, video, text_input, upload_button, chat_state, img_list], queue=False)
    
demo.launch(share=False, enable_queue=True)


# %%
