"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import json

import numpy as np
import torch
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation
import decord
decord.bridge.set_bridge('torch')

import pytube

def patch_pytube(pt):
    import xml.etree.ElementTree as ElementTree
    from html import unescape

    def xml_caption_to_srt(self, xml_captions: str) -> str:
        """Convert xml caption tracks to "SubRip Subtitle (srt)".

        :param str xml_captions:
            XML formatted caption tracks.
        """
        segments = []
        root = ElementTree.fromstring(xml_captions)[0]
        i = 0
        for child in list(root):
            if child.tag == 'p':
                caption = child.text
                caption = unescape(caption.replace(
                    "\n", " ").replace("  ", " "),)
                try:
                    duration = float(child.attrib["d"])/1000.0
                except KeyError:
                    duration = 0.0
                start = float(child.attrib["t"])/1000.0
                end = start + duration
                sequence_number = i + 1  # convert from 0-indexed to 1.
                line = "{seq}\n{start} --> {end}\n{text}\n".format(
                    seq=sequence_number,
                    start=self.float_to_srt_time_format(start),
                    end=self.float_to_srt_time_format(end),
                    text=caption,
                )
                segments.append(line)
                i += 1
        return "\n".join(segments).strip()
    pt.captions.Caption.xml_caption_to_srt = xml_caption_to_srt

    def json_captions(self):
        import json
        from pytube import request
        json_captions_url = self.url
        if 'fmt=srv3' in self.url:
            json_captions_url = self.url.replace('fmt=srv3', 'fmt=json3')
        else:
            json_captions_url += '&fmt=json3'
        text = request.get(json_captions_url)
        parsed = json.loads(text)
        assert parsed['wireMagic'] == 'pb3', 'Unexpected captions format'
        return parsed

    setattr(pt.captions.Caption, 'json_captions', property(json_captions))

    def vid_info(self):
        """Parse the raw vid info and return the parsed result.

        :rtype: Dict[Any, Any]
        """
        if self._vid_info:
            return self._vid_info

        innertube = pt.innertube.InnerTube(
            use_oauth=self.use_oauth, allow_cache=self.allow_oauth_cache, client='WEB')

        innertube_response = innertube.player(self.video_id)
        self._vid_info = innertube_response
        return self._vid_info
    setattr(pt.YouTube, 'vid_info', property(vid_info))
    
    # Patch from: https://github.com/24makee/pytube/blob/bugfix/bug-1702/pytube/contrib/channel.py
    import logging
    from pytube.helpers import uniqueify
    def _extract_videos(raw_json):
        """Extracts videos from a raw json page

        :param str raw_json: Input json extracted from the page or the last
            server response
        :rtype: Tuple[List[str], Optional[str]]
        :returns: Tuple containing a list of up to 100 video watch ids and
            a continuation token, if more videos are available
        """
        initial_data = json.loads(raw_json)
        # this is the json tree structure, if the json was extracted from
        # html
        try:
            videos = initial_data["contents"][
                "twoColumnBrowseResultsRenderer"][
                "tabs"][1]["tabRenderer"]["content"][
                "richGridRenderer"]["contents"]
        except (KeyError, IndexError, TypeError):
            try:
                # this is the json tree structure, if the json was directly sent
                # by the server in a continuation response
                important_content = initial_data[1]['response']['onResponseReceivedActions'][
                    0
                ]['appendContinuationItemsAction']['continuationItems']
                videos = important_content
            except (KeyError, IndexError, TypeError):
                try:
                    # this is the json tree structure, if the json was directly sent
                    # by the server in a continuation response
                    # no longer a list and no longer has the "response" key
                    important_content = initial_data['onResponseReceivedActions'][0][
                        'appendContinuationItemsAction']['continuationItems']
                    videos = important_content
                except (KeyError, IndexError, TypeError) as p:
                    logger.info(p)
                    return [], None

        try:
            continuation = videos[-1]['continuationItemRenderer'][
                'continuationEndpoint'
            ]['continuationCommand']['token']
            videos = videos[:-1]
        except (KeyError, IndexError):
            # if there is an error, no continuation is available
            continuation = None

        # remove duplicates
        return (
            uniqueify(
                list(
                    # only extract the video ids from the video data
                    map(
                        lambda x: (
                            f"/watch?v="
                            f"{x['richItemRenderer']['content']['videoRenderer']['videoId']}"
                        ),
                        videos
                    )
                ),
            ),
            continuation,
        )    
    setattr(pt.contrib.channel.Channel, '_extract_videos', staticmethod(_extract_videos))
    
    return pt


pytube = patch_pytube(pytube)


#%%
from video_llama.models import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
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

def convert_to_minutes_seconds(start_time_ms, duration_ms):
    # Convert start time and end time to seconds
    start_seconds = start_time_ms / 1000
    end_seconds = (start_time_ms + duration_ms) / 1000

    # Calculate minutes and seconds for start time
    start_minutes = int(start_seconds // 60)
    start_seconds = int(start_seconds % 60)

    # Calculate minutes and seconds for end time
    end_minutes = int(end_seconds // 60)
    end_seconds = int(end_seconds % 60)

    # Format the time range
    formatted_time_range = f"[{start_minutes:02d}:{start_seconds:02d} - {end_minutes:02d}:{end_seconds:02d}]"
    
    return formatted_time_range

def parse_captions(caption):
    # print(json.dumps(caption.json_captions, indent=4))
    # exit()    
    if caption is None:
        return ''
    all_segs = []
    for e in caption.json_captions['events']:
        if 'segs' not in e or len(e['segs']) == 0:
            continue
        start_time_ms = e["tStartMs"]
        if "dDurationMs" not in e or ('aAppend' in e and e['aAppend']):
            all_segs.extend(['\n'])
            continue
        duration_ms = e["dDurationMs"]
        caption_time_str = convert_to_minutes_seconds(start_time_ms, duration_ms) + '  '
        all_segs.extend([caption_time_str] + [ee['utf8'] for ee in e['segs']])
    # all_segs = [' ' if c == '\n' else c for c in all_segs]
    captions_str = ''.join(all_segs)
    #print('Captions: ', captions_str)
    return captions_str


def select_en_captions(captions):
    for c in captions:
        if 'en' in c.code:
            return c
    return None


def get_transcript(video):
    return parse_captions(select_en_captions(video.captions))

def download_yt_video(video_url, use_transcripts=True):
    yt_video = pytube.YouTube(video_url)
    video_id = yt_video.video_id
    saved_video_path = f'/tmp/saved_videos/{video_id}.mp4'
    if not os.path.exists(os.path.dirname(saved_video_path)):
        os.makedirs(os.path.dirname(saved_video_path))
    print('downloading video')
    if not os.path.exists(saved_video_path):
        saved_video_path = yt_video.streams.filter(only_video=True).order_by('resolution').asc().first().download(output_path=os.path.dirname(saved_video_path), filename=os.path.basename(saved_video_path))
    print('downloaded video')
    print(saved_video_path)
    print('downloading transcripts')
    if use_transcripts:
        transcript = get_transcript(yt_video)
        saved_transcript_path = f'/tmp/saved_videos/{video_id}.json'
        json.dump(transcript, open(saved_transcript_path, 'w'), )
        return saved_video_path, saved_transcript_path
    print('downloaded transcripts')
    return saved_video_path, None

def upload_video(gr_video, text_input, chat_state, chatbot, use_transcripts, transcript_path):
    if use_transcripts:
        transcript = json.load(open(transcript_path))
        # TODO: Add transcript to video.
        print(transcript)
    chat_state = default_conversation.copy()
    if gr_video is not None:
        print(gr_video)
        chatbot = chatbot + [((gr_video,), None)]
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list, num_frames=16)
        if use_transcripts:
            chat_state.messages[0][1] += f'\n Transcript: {transcript}\n'
        #print(chat_state.messages)
        #print(chat_state.messages[0][1])
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
            #video = gr.Video()
            video_url = gr.Textbox(label='VideoURL', placeholder='Enter YouTube video URL', interactive=True)

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
            use_transcripts = gr.Checkbox(interactive=True, value=False, label="Use transcripts?")

        with gr.Column():
            video = gr.State()
            transcript = gr.State()
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='Ask-Videos')
            text_input = gr.Textbox(label='User', placeholder='Upload your video first, or directly click the examples at the bottom of the page.', interactive=False)
            
        
    gr.Markdown(cite_markdown)
    upload_button.click(download_yt_video, [video_url, use_transcripts], [video, transcript]).then(upload_video, [video, text_input, chat_state, chatbot, use_transcripts, transcript], [video, text_input, upload_button, chat_state, img_list, chatbot])
    #upload_button.click(upload_video, [video, text_input, chat_state, chatbot], [video, text_input, upload_button, chat_state, img_list, chatbot])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, video, text_input, upload_button, chat_state, img_list], queue=False)
    
demo.launch(share=False, enable_queue=True)


# %%
