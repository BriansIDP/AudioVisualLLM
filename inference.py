from operator import truediv
from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
import os
import time
import ipdb
import mdtex2html
from model.openllama import OpenLLAMAPEFTModel
import torch
import json


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(
    input, 
    image_path, 
    audio_path, 
    video_path, 
    thermal_path, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache,
):
    if image_path is None and audio_path is None and video_path is None and thermal_path is None:
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]

    # prepare the prompt
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\nASSISTANT: {a}\n'
        else:
            prompt_text += f' USER: {q}\nASSISTANT: {a}\n'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' USER: {input}'
    inputs = {
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'dosample': True,
        'modality_embeds': modality_cache,
        'av_shift': False,
        'num_beams': 3,
        'num_returns': 3,
        'lengthpenalty': 1.0,
    }
    response, _ = model.generate(inputs)
    history.append((input, response[0]))
    return response, history, modality_cache


if __name__ == "__main__":
    # init the model
    expname = "audiovisual_vicuna13b_sepqformer_avsd_earlyalign_swqformer_causal_tune"
    # expname="audiovisual_vicuna7b_sepqformer_avsd_earlyalign_swqformer_causal_diversity"
    args = {
        'model': 'openllama_peft',
        'imagebind_ckpt_path': "",
        'vicuna_ckpt_path': "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/llama/vicuna.13b/",
        'orig_delta_path': "", #'../pretrained_ckpt/pandagpt_ckpt/13b/pytorch_model.pt',
        'delta_ckpt_path': f"ckpt/{expname}/pytorch_model_1_101.pt",
        'stage': 2,
        'max_tgt_len': 256,
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'use_lora': "true",
        'qformer': "true",
        'use_whisper': "true",
        'use_blip': "true",
        'instructblip': "true",
        'proj_checkpoint': "",
        'num_video_query': 32,
        'num_speech_query': 32,
        'instructblip_video': "false",
        'video_window_size': 240,
        'skip_vqformer': "false",
        'speech_qformer': "false",
        'early_align': "true",
        'cascaded': "",
        'causal': "false",
        'diversity_loss': "false",
        'causal_attention': "true",
        'modalitymask': 'false',
        'groupsize': 10,
        'alignmode': 2,
    }
    model = OpenLLAMAPEFTModel(**args)
    if args['orig_delta_path'] != '':
        orig_ckpt = torch.load(args['orig_delta_path'], map_location=torch.device('cpu'))
        model.load_state_dict(orig_ckpt, strict=False)
    delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().cuda()
    print(f'[!] init the 13b model over ...')
    modality_cache = []

    testimage = False
    withaudio = True
    multiturn = False
    video_path = "data/example_video.mp4"
    audio_path = "data/example_video.wav"
    user_input = "Explain in detail why this video together with the audio and what they say is romantic"
    response, history, modality_cache = predict(user_input, None, audio_path, video_path, None, 512, 0.9, 1, [], [])
    print(response[0])
