from pickle import TRUE
from header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .modeling_llama import LlamaForCausalLM, LlamaConfig
from .Qformer import BertConfig, BertLMHeadModel
from .modeling_whisper import WhisperModel
from .eva_vit import create_eva_vit_g
from transformers import StoppingCriteria, StoppingCriteriaList, BertTokenizer
from transformers import WhisperFeatureExtractor
import soundfile as sf

import torch
from torch.nn.utils import rnn

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = '</Img> ' + turn['value'] + '\nASSISTANT:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'USER: ' + turn['value'] + '\nASSISTANT:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                # text = turn['value'] + '\n###'
                text = turn['value'] + '</s>'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len, modality='image'):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        text_list, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

PROMPT_START = 'USER: <Img>'
dummy_image_path = ["dummy/761183272.jpg"]
dummy_audio_path = ["dummy/1272-128104-0000.flac"]

class OpenLLAMAPEFTModel(nn.Module):

    '''LoRA for LLaMa model'''

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args['imagebind_ckpt_path']
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        max_tgt_len = args['max_tgt_len']
        stage = args['stage']
        self.use_whisper = (args["use_whisper"] == "true")
        self.use_blip = (args["use_blip"] == "true")
        self.instructblip = (args["instructblip"] == "true")
        self.instructblip_video = (args["instructblip_video"] == "true")
        self.skip_vqformer = (args["skip_vqformer"] == "true")
        self.video_window_size = args["video_window_size"]
        self.speech_qformer = (args["speech_qformer"] == "true")
        self.early_align = (args["early_align"] == "true")
        self.cascaded = args["cascaded"]
        self.causal = (args["causal"] == "true")
        self.diversity = (args["diversity_loss"] == "true")
        self.diversity_loss_factor = args.get("diversity_loss_factor", 0.01)
        self.causal_encoder = (args.get("causal_attention", "false") == "true")
        self.groupsize = args.get("groupsize", 0)
        self.alignmode = args.get("alignmode", 1)
        self.modalitymask = (args.get("modalitymask", "false") == "true")
        self.xsegalign = (args.get("xsegalign", "false") == "true")
        self.seglen = args.get("seglen", 3)

        self.cache_dir = "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/opensource/favor/pretrained_ckpt/cache"

        if self.use_blip:
            print("Loading visual encoder ViT")
            self.visual_encoder = create_eva_vit_g(
                224, 0, False, "fp16"
            )
            print("Finished loading visual encoder ViT")
            self.ln_vision = nn.LayerNorm(self.visual_encoder.num_features)
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            print("Loading Qformer")
            self.num_query_token = 32
            self.Qformer, self.query_tokens = self.init_video_Qformer(
                num_query_token = self.num_query_token,
                vision_width=self.visual_encoder.num_features,
                num_hidden_layers = -1,
            )
            if self.instructblip:
                self.bert_tokenizer = BertTokenizer.from_pretrained(
                    "bert-base-uncased",
                    truncation_side="left",
                    cache_dir=self.cache_dir,
                )
                self.bert_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
                self.Qformer.resize_token_embeddings(len(self.bert_tokenizer))
            else:
                self.Qformer.bert.embeddings.word_embeddings = None
                self.Qformer.bert.embeddings.position_embeddings = None
                for layer in self.Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
            self.Qformer.cls = None

            # delete later
            if self.instructblip:
                # self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, 5120)
                qformer_path = "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/opensource/favor/pretrained_ckpt/instruct_blip_vicuna13b_trimmed.pth"
            else:
                qformer_path = "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/opensource/favor"
            state_dict = torch.load(qformer_path, map_location="cpu")["model"]
            msg = self.load_state_dict(state_dict, strict=False)
            # print(f"Finished loading Qformer, Unused parameters: {msg}")
            # Freeze Qformer
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.query_tokens.requires_grad = False
            self.visual_hidden_size = self.Qformer.config.hidden_size
        else:
            print (f'Initializing visual encoder from {imagebind_ckpt_path} ...')
            self.visual_encoder, self.visual_hidden_size = \
            imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
            # freeze vision encoder
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
            print ('Visual encoder initialized.')

        if self.use_whisper:
            print("Loading Whisper Model")
            whispermodel = "openai/whisper-large-v2"
            self.whispertransform = WhisperFeatureExtractor.from_pretrained(whispermodel, cache_dir=self.cache_dir)
            self.speech_encoder = WhisperModel.from_pretrained(whispermodel).encoder
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder = self.speech_encoder.eval()
            print("Freeze Whisper and Loading Whisper done.")

        print (f'Initializing language decoder from {vicuna_ckpt_path} ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
        if self.args['use_lora'] == 'true':
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        else:
            print("Not updating vicuna at all!")
            # free vicuna
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print ('Language decoder initialized.')

        self.use_qformer = args["qformer"] if "qformer" in args else "false"
        if self.use_qformer == 'true':
            if self.use_whisper and not self.speech_qformer:
                self.speech_pre_qformer_proj = nn.Linear(self.speech_encoder.config.d_model, self.visual_hidden_size)
            if self.early_align and self.alignmode == 2:
                if self.speech_qformer:
                    self.visual_hidden_size = self.visual_hidden_size + self.speech_encoder.config.d_model
                else:
                    self.visual_hidden_size = self.visual_hidden_size * 2
            self.ln_video = nn.LayerNorm(self.visual_hidden_size)
            # if not self.use_whisper:
            self.video_frame_position_embedding = nn.Embedding(4096, self.visual_hidden_size)
            self.num_video_query_token = args['num_video_query']
            self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(
                num_query_token = self.num_video_query_token,
                vision_width=self.visual_hidden_size,
                num_hidden_layers = self.seglen if self.xsegalign else 2,
                causal_encoder=self.causal_encoder,
            )
            if self.instructblip_video:
                self.video_Qformer.resize_token_embeddings(len(self.bert_tokenizer))
                Qformer_embeddings = self.Qformer.bert.embeddings.state_dict()
                self.video_Qformer.bert.embeddings.load_state_dict(Qformer_embeddings)
            else:
                self.video_Qformer.bert.embeddings.word_embeddings = None
                self.video_Qformer.bert.embeddings.position_embeddings = None
                if not self.xsegalign:
                    for layer in self.video_Qformer.bert.encoder.layer:
                        layer.output = None
                        layer.intermediate = None
            self.video_Qformer.cls = None
            self.llama_proj = nn.Linear(
                self.video_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )

            if self.speech_qformer:
                # A separate speech Qformer
                self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
                self.speech_Qformer, self.speech_query_tokens = self.init_video_Qformer(
                    num_query_token = args.get("num_speech_query", self.num_video_query_token),
                    vision_width=self.speech_encoder.config.d_model,
                    num_hidden_layers = 2,
                    causal_encoder=self.causal_encoder,
                )
                if self.instructblip_video:
                    self.speech_Qformer.resize_token_embeddings(len(self.bert_tokenizer))
                else:
                    self.speech_Qformer.bert.embeddings.word_embeddings = None
                    self.speech_Qformer.bert.embeddings.position_embeddings = None
                    for layer in self.speech_Qformer.bert.encoder.layer:
                        layer.output = None
                        layer.intermediate = None
                self.speech_Qformer.cls = None
                self.llama_proj_speech = nn.Linear(
                    self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size
                )
            else:
                self.speech_Qformer = self.video_Qformer
                self.speech_query_tokens = self.video_query_tokens
                self.llama_proj_speech = self.llama_proj
                self.ln_speech = self.ln_video

            if self.early_align and self.alignmode == 3:
                # A separate speech Qformer
                self.speech_pre_qformer_proj = nn.Linear(self.speech_encoder.config.d_model, self.visual_hidden_size)
                self.joint_size = self.visual_hidden_size * 2
                self.joint_frame_position_embedding = nn.Embedding(4096, self.joint_size)
                self.ln_joint = nn.LayerNorm(self.joint_size)
                self.joint_Qformer, self.joint_query_tokens = self.init_video_Qformer(
                    num_query_token = self.num_video_query_token,
                    vision_width=self.joint_size,
                    num_hidden_layers = 2,
                    causal_encoder=self.causal_encoder,
                )
                self.joint_Qformer.bert.embeddings.word_embeddings = None
                self.joint_Qformer.bert.embeddings.position_embeddings = None
                for layer in self.joint_Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
                self.joint_Qformer.cls = None
                self.llama_proj_joint = nn.Linear(
                    self.joint_Qformer.config.hidden_size, self.llama_model.config.hidden_size
                )
            elif self.early_align and self.alignmode == 2:
                self.joint_Qformer = self.video_Qformer
                self.joint_query_tokens = self.video_query_tokens
                self.llama_proj_joint = self.llama_proj
                self.ln_joint = self.ln_video
                self.joint_frame_position_embedding = self.video_frame_position_embedding
            # if self.instructblip:
            #     self.llama_proj.load_state_dict(self.llm_proj.state_dict())
        else:
            self.llama_proj = nn.Linear(
                self.visual_hidden_size, self.llama_model.config.hidden_size
            )

        if args["proj_checkpoint"] != "":
            proj_state = torch.load(args["proj_checkpoint"])
            msg = self.load_state_dict(proj_state, strict=False)

        self.max_tgt_len = max_tgt_len
        self.device = torch.cuda.current_device()
        if args['delta_ckpt_path']:
            delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=self.llama_model.device)
            self.load_state_dict(delta_ckpt, strict=False)

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, 
                           num_hidden_layers=2, causal_encoder=False):
        encoder_config = BertConfig.from_pretrained(
            "bert-base-uncased",
            cache_dir="/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/opensource/favor/pretrained_ckpt/cache",
        )
        if num_hidden_layers > 0:
            encoder_config.num_hidden_layers = num_hidden_layers
            encoder_config.cross_attention_freq = 1
            encoder_config.causal_encoder = causal_encoder
        else:
            encoder_config.cross_attention_freq = 2
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def encode_video(self, video_paths, instruction_inds=None, instruction_embs=None, earlyalign=False, audio_query=None):
        if self.use_blip:
            inputs, video_masks = data.load_and_transform_video_data_blip(video_paths, self.device)
            bsize, nframes = inputs.size(0), inputs.size(1)
            inputs = inputs.to(self.llama_model.dtype).view(
                bsize * nframes, inputs.size(2), inputs.size(3), inputs.size(4))
            with torch.no_grad():
                video_embeds = self.ln_vision(self.visual_encoder(inputs))
                video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video_embeds.device)
                query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)

                if self.instructblip and instruction_inds is not None:
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(video_embeds.device)
                    instruction_mask = instruction_inds.attention_mask.unsqueeze(1).repeat(1, nframes, 1).view(bsize * nframes, -1)
                    Qformer_atts = torch.cat([query_atts, instruction_mask], dim=1)
                    input_ids = instruction_inds.input_ids.unsqueeze(1).repeat(1, nframes, 1).view(bsize * nframes, -1)
                    query_output = self.Qformer.bert(
                        input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=video_embeds,
                        encoder_attention_mask=video_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=video_embeds,
                        encoder_attention_mask=video_atts,
                        return_dict=True,
                    )
                video_embeds = query_output.last_hidden_state  # (B * T) * Q * H
                if self.instructblip and instruction_inds is not None:
                    video_embeds = video_embeds[:, :self.num_query_token]
                # delete later
                # orig_video_embeds = video_embeds.reshape(bsize, nframes, self.num_query_token, video_embeds.size(-1))
                # sel_index = torch.linspace(0, nframes-1, steps=5).long().to(self.device)
                # orig_video_embeds = orig_video_embeds[torch.arange(bsize), sel_index]
                # orig_video_embeds = orig_video_embeds.reshape(bsize, 5*self.num_query_token, video_embeds.size(-1))

                video_embeds = video_embeds.reshape(bsize, nframes * self.num_query_token, video_embeds.size(-1))
                video_masks = video_masks.unsqueeze(-1).repeat(1, 1, self.num_query_token).view(bsize, -1)
                if self.video_window_size < nframes:
                    pad_len = (nframes // self.video_window_size + 1) * self.video_window_size - nframes
                    pad_len = int(pad_len * self.num_query_token)
                    n_windows = int(nframes // self.video_window_size + 1)
                    pad_video_embeds = video_embeds.new_zeros(bsize, pad_len, video_embeds.size(-1))
                    pad_video_masks = video_masks.new_zeros(bsize, pad_len)
                    video_embeds = torch.cat([video_embeds, pad_video_embeds], dim=1).view(
                        bsize * n_windows, -1, video_embeds.size(-1))
                    video_masks = torch.cat([video_masks, pad_video_masks], dim=1).view(
                        bsize * n_windows, -1)
        else:
            video_features, video_masks = data.load_and_transform_video_data_full(video_paths, self.device)
            inputs = {ModalityType.VISION: video_features}
            # convert into visual dtype
            inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)
                video_embeds = embeddings[ModalityType.VISION] # bsz x T x 1024

        if earlyalign and self.alignmode != 3:
            return video_embeds, video_masks, video_embeds
        elif earlyalign and self.alignmode == 3:
            pre_qformer_embeds = video_embeds

        if self.use_qformer == 'true':
            video_embeds = self.ln_video(video_embeds)
            position_ids = torch.arange(video_embeds.size(1), dtype=torch.long, device=video_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(video_embeds.size(0), -1)
            # print(video_embeds.size(1))
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_hidden_state = frame_position_embeddings + video_embeds
            frame_atts = video_masks.long()
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            if audio_query is not None:
                video_query_tokens = torch.cat([video_query_tokens, audio_query], dim=1)
            if self.instructblip_video and instruction_inds is not None:
                query_atts = torch.ones(video_query_tokens.size()[:-1], dtype=torch.long).to(video_embeds.device)
                if self.video_window_size < nframes:
                    instruction_mask = instruction_inds.attention_mask.unsqueeze(1).repeat(
                        1, n_windows, 1).view(bsize * n_windows, -1)
                    input_ids = instruction_inds.input_ids.unsqueeze(1).repeat(
                        1, n_windows, 1).view(bsize * n_windows, -1)
                else:
                    instruction_mask = instruction_inds.attention_mask
                    input_ids = instruction_inds.input_ids
                Qformer_atts = torch.cat([query_atts, instruction_mask], dim=1)
                video_query_output = self.video_Qformer.bert(
                    input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
                # video_query_tokens = torch.cat([video_query_tokens, instruction_embs], dim=1)
            else:
                video_query_output = self.video_Qformer.bert(
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
            video_embeds = video_query_output.last_hidden_state
            video_embeds = video_embeds[:, :self.num_video_query_token]
            if self.video_window_size < nframes:
                video_embeds = video_embeds.view(bsize, -1, video_embeds.size(-1)) # bsz x Q*n_windows x embsize
                inputs_llama = self.llama_proj(video_embeds) # bsz x Q x llama_size
                atts_llama = (video_masks.sum(dim=-1) != 0).unsqueeze(1).repeat(
                    1, self.num_query_token).view(bsize, -1)
            else:
                inputs_llama = self.llama_proj(video_embeds) # bsz x Q x llama_size
                # delete later
                # inputs_llama = self.llm_proj(orig_video_embeds)
                atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        else:
            atts_llama = video_masks.long()
            inputs_llama = self.llama_proj(video_embeds) # bsz x T x llama_size
        if earlyalign and self.alignmode == 3:
            return pre_qformer_embeds, atts_llama, inputs_llama
        else:
            return inputs_llama, atts_llama, video_embeds

    def encode_audio(self, audio_paths, instruction_inds=None, earlyalign=False, visual_query=None):
        if self.use_whisper:
            # For test only
            if len(audio_paths) == 1 and isinstance(audio_paths[0], str):
                audio, _ = sf.read(audio_paths[0])
                if len(audio.shape) == 2:
                    audio = audio[:, 0]
                spectrogram = self.whispertransform(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    max_length=30 * 16000,
                )
                audio_paths = [spectrogram["input_features"].squeeze()]

            with torch.no_grad():
                audio_paths = torch.stack(audio_paths, dim=0).to(self.device)
                audio_embeds = self.speech_encoder(
                    audio_paths.to(self.llama_model.dtype), return_dict=True).last_hidden_state
            if not self.speech_qformer:
                audio_embeds = self.speech_pre_qformer_proj(audio_embeds)
        else:
            inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data_fulllen(audio_paths, self.device)}
            # convert into visual dtype
            inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)
                audio_embeds = embeddings[ModalityType.AUDIO] # bsz x T x 1024

        if earlyalign and self.alignmode != 3:
            return audio_embeds, None
        elif earlyalign and self.alignmode == 3:
            pre_qformer_embed = audio_embeds

        if self.use_qformer == 'true':
            audio_embeds = self.ln_speech(audio_embeds)
            position_ids = torch.arange(audio_embeds.size(1), dtype=torch.long, device=audio_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(audio_embeds.size(0), -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids).mean() * 0
            frame_hidden_state = frame_position_embeddings + audio_embeds  # audio/speech do not use pos enc
            frame_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio_embeds.device)
            # frame_atts = video_masks.long()
            speech_query_tokens = self.speech_query_tokens.expand(audio_embeds.shape[0], -1, -1)
            if visual_query is not None:
                speech_query_tokens = torch.cat([speech_query_tokens, visual_query], dim=1)
            if self.instructblip_video and instruction_inds is not None:
                query_atts = torch.ones(speech_query_tokens.size()[:-1], dtype=torch.long).to(audio_embeds.device)
                Qformer_atts = torch.cat([query_atts, instruction_inds.attention_mask],dim=1)
                audio_query_output = self.speech_Qformer.bert(
                    instruction_inds.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=speech_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
                # video_query_tokens = torch.cat([video_query_tokens, instruction_embs], dim=1)
            else:
                audio_query_output = self.speech_Qformer.bert(
                    query_embeds=speech_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
            audio_embeds = audio_query_output.last_hidden_state
            audio_embeds = audio_embeds[:, :self.num_video_query_token]
            inputs_llama = self.llama_proj_speech(audio_embeds) # bsz x Q x llama_size
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        else:
            # atts_llama = video_masks.long()
            inputs_llama = self.llama_proj(audio_embeds) # bsz x T x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        if earlyalign and self.alignmode == 3:
            return pre_qformer_embed, inputs_llama
        else:
            return inputs_llama, audio_embeds

    def encode_image(self, image_paths, instruction_inds=None, instruction_embs=None, earlyalign=False, audio_query=None):
        if self.use_blip:
            inputs = data.load_and_transform_vision_data_blip(image_paths, self.device, self.training)
            inputs = inputs.to(self.llama_model.dtype)
            with torch.no_grad():
                image_embeds = self.ln_vision(self.visual_encoder(inputs))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

                if self.instructblip and instruction_inds is not None:
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
                    Qformer_atts = torch.cat([query_atts, instruction_inds.attention_mask],dim=1)
                    query_output = self.Qformer.bert(
                        instruction_inds.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                image_embeds = query_output.last_hidden_state  # bsz x 32 x H
                if self.instructblip and instruction_inds is not None:
                    image_embeds = image_embeds[:, :self.num_query_token]
        else:
            inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
            # convert into visual dtype
            inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)
                image_embeds = embeddings['vision'].unsqueeze(1)  # bsz x 1 x 1024

        if earlyalign and self.alignmode != 3:
            return image_embeds, None
        elif earlyalign and self.alignmode == 3:
            pre_qformer_embed = image_embeds

        if self.use_qformer == 'true':
            orig_image_embeds = image_embeds
            image_embeds = self.ln_video(image_embeds)
            position_ids = torch.arange(image_embeds.size(1), dtype=torch.long, device=image_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(image_embeds.size(0), -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_hidden_state = frame_position_embeddings + image_embeds
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(image_embeds.device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            if audio_query is not None:
                video_query_tokens = torch.cat([video_query_tokens, audio_query], dim=1)
            if self.instructblip_video and instruction_inds is not None:
                query_atts = torch.ones(video_query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
                Qformer_atts = torch.cat([query_atts, instruction_inds.attention_mask],dim=1)
                video_query_output = self.video_Qformer.bert(
                    instruction_inds.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
                # video_query_tokens = torch.cat([video_query_tokens, instruction_embs], dim=1)
            else:
                video_query_output = self.video_Qformer.bert(
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
            image_embeds = video_query_output.last_hidden_state
            image_embeds = image_embeds[:, :self.num_video_query_token]
            if self.skip_vqformer:
                image_embeds = image_embeds * 0 + orig_image_embeds

        inputs_llama = self.llama_proj(image_embeds) # bsz x 1 x llama_size
        # delete later
        # inputs_llama = self.llm_proj(orig_image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        if earlyalign and self.alignmode == 3:
            return pre_qformer_embed, inputs_llama
        else:
            return inputs_llama, image_embeds

    def sequence_align_v2(self, video_embeds, video_masks, audio_embeds, inputmasks=None, instruction_inds=None):
        """Args:
        video_embeds: B x 32 T1 x D
        video_mask: B x 32 T1
        audio_embeds: B x T1 x D
        inputsmasks: B x 2
        """
        video_size = 32
        audio_size = 25
        bsize = video_embeds.size(0)
        vid_T = video_embeds.size(1) // video_size
        video_embeds = video_embeds.view(bsize, -1, video_size, video_embeds.size(-1))
        audio_embeds = audio_embeds.view(bsize, -1, audio_size, audio_embeds.size(-1))
        if self.early_align and self.alignmode == 3:
            audio_embeds = self.speech_pre_qformer_proj(audio_embeds)
        aud_T = audio_embeds.size(1)
        if aud_T > vid_T:
            diff_T = aud_T - vid_T
            video_pad = video_embeds.new_zeros(bsize, diff_T, video_size, video_embeds.size(-1))
            video_embeds = torch.cat([video_embeds, video_pad], dim=1)
        elif aud_T < vid_T:
            diff_T = vid_T - aud_T
            audio_pad = audio_embeds.new_zeros(bsize, diff_T, audio_size, audio_embeds.size(-1))
            audio_embeds = torch.cat([audio_embeds, audio_pad], dim=1)
        if video_size - audio_size > 0:
            audio_token_padding = audio_embeds.new_zeros(bsize, audio_embeds.size(1), video_size - audio_size, audio_embeds.size(-1))
            audio_embeds = torch.cat([audio_embeds, audio_token_padding], dim=2)
        elif video_size - audio_size < 0:
            video_token_padding = video_embeds.new_zeros(bsize, video_embeds.size(1), audio_size - video_size, video_embeds.size(-1))
            video_embeds =  torch.cat([video_embeds, video_token_padding], dim=2)
        if inputmasks is not None:
            video_embeds = video_embeds * inputmasks[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            audio_embeds = audio_embeds * inputmasks[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        maxsize = max(video_size, audio_size)
        concat_features = torch.cat([video_embeds, audio_embeds], dim=3).view(
            bsize, video_embeds.size(1) * maxsize, -1)
        total_mask = concat_features.new_ones(concat_features.size()[:-1])

        if self.groupsize >= 1:
            ngroups = video_embeds.size(1) // self.groupsize
            if video_embeds.size(1) % self.groupsize != 0:
                ngroups = ngroups + 1
                diff_T = ngroups * self.groupsize - video_embeds.size(1)
                concat_feature_paddings = concat_features.new_zeros(bsize, diff_T * maxsize, concat_features.size(-1))
                concat_features = torch.cat([concat_features, concat_feature_paddings], dim=1)
                total_mask = torch.cat([total_mask, total_mask.new_zeros(bsize, diff_T * maxsize)], dim=1)
            concat_features = concat_features.view(bsize * ngroups, self.groupsize * maxsize, concat_features.size(-1))
            total_mask = total_mask.view(bsize * ngroups, self.groupsize * maxsize)

        # Forward Q-Former
        total_embeds = self.ln_joint(concat_features)  # B x 32*T_max x D
        position_ids = torch.arange(total_embeds.size(1), dtype=torch.long, device=total_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(total_embeds.size(0), -1)
        frame_position_embeddings = self.joint_frame_position_embedding(position_ids)
        frame_hidden_state = frame_position_embeddings + total_embeds
        joint_query_tokens = self.joint_query_tokens.expand(total_embeds.shape[0], -1, -1)

        av_query_output = self.joint_Qformer.bert(
            query_embeds=joint_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=total_mask,
            return_dict=True,
            segsize=ngroups if self.xsegalign else 0,
        )
        total_embeds = av_query_output.last_hidden_state
        if self.xsegalign and ngroups > self.seglen:
            total_embeds = total_embeds[:, :self.num_video_query_token].reshape(bsize, ngroups, -1, total_embeds.size(-1))
            sel_index = torch.linspace(self.seglen-1, ngroups-1, steps=ngroups//self.seglen + 1).long().to(self.device)
            total_embeds = total_embeds[:, sel_index]
        inputs_llama = self.llama_proj_joint(total_embeds)
        if self.groupsize > 1:
            inputs_llama = inputs_llama.reshape(bsize, -1, inputs_llama.size(-1))
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, total_embeds

    def sequence_align(self, video_embeds, video_mask, audio_embeds, inputmasks=None, instruction_inds=None):
        """Args:
        video_embeds: B x 32 T1 x D
        video_mask: B x 32 T1
        audio_embeds: B x T1 x D
        inputsmasks: B x 2
        """
        video_size = 32 * self.groupsize
        audio_size = 25 * self.groupsize
        bsize = video_embeds.size(0)
        vid_T = video_embeds.size(1) // video_size
        vid_T += 1 if video_embeds.size(1) % video_size != 0 else 0
        aud_T = audio_embeds.size(1) // audio_size
        aud_T += 1 if video_embeds.size(1) % video_size != 0 else 0
        max_T = max(vid_T, aud_T)
        audio_mask = video_mask.new_ones(audio_embeds.size()[:-1])
        if video_embeds.size(1) < max_T * video_size:
            padlen = max_T * video_size - video_embeds.size(1)
            video_pad = video_embeds.new_zeros(bsize, padlen, video_embeds.size(-1))
            video_mask_pad = video_mask.new_zeros(bsize, padlen)
            video_embeds = torch.cat([video_embeds, video_pad], dim=1)
            video_mask = torch.cat([video_mask, video_mask_pad], dim=1)
        if audio_embeds.size(1) < max_T * audio_size:
            padlen = max_T * audio_size - audio_embeds.size(1)
            audio_pad = audio_embeds.new_zeros(bsize, padlen, audio_embeds.size(-1))
            audio_mask_pad = audio_mask.new_zeros(bsize, padlen)
            audio_embeds = torch.cat([audio_embeds, audio_pad], dim=1)
            audio_mask = torch.cat([audio_mask, audio_mask_pad], dim=1)
        if inputmasks is not None:
            video_embeds = video_embeds * inputmasks[:, 0:1].unsqueeze(-1)
            audio_embeds = audio_embeds * inputmasks[:, 1:2].unsqueeze(-1)
        video_embeds = video_embeds.view(-1, video_size, video_embeds.size(-1))
        audio_embeds = audio_embeds.view(-1, audio_size, audio_embeds.size(-1))
        video_mask = video_mask.view(-1, video_size)
        audio_mask = audio_mask.view(-1, audio_size)

        video_embeds = self.ln_video(video_embeds)
        audio_embeds = self.ln_speech(audio_embeds)
        position_ids = torch.arange(video_size, dtype=torch.long, device=video_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(video_embeds.size(0), -1)
        frame_position_embeddings = self.video_frame_position_embedding(position_ids)
        frame_hidden_state = frame_position_embeddings + video_embeds
        video_query_tokens = self.video_query_tokens.expand(video_embeds.size(0), -1, -1)
        audio_query_tokens = self.speech_query_tokens.expand(audio_embeds.size(0), -1, -1)
        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=video_mask,
            return_dict=True,
        )
        video_query_output = video_query_output.last_hidden_state
        audio_query_output = self.speech_Qformer.bert(
            query_embeds=audio_query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_mask,
            return_dict=True,
        )
        audio_query_output = audio_query_output.last_hidden_state

        total_embeds = torch.cat([video_query_output, audio_query_output], dim=1)
        inputs_llama = self.llama_proj(total_embeds) # B*T_max x Q x llama_size
        inputs_llama = inputs_llama.view(bsize, -1, inputs_llama.size(-1))  # B x T_max*Q x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, video_query_output

    def prompt_wrap(
        self,
        img_embeds,
        input_ids,
        target_ids,
        attention_mask,
        audio_embs=None,
        audiomasks=None,
    ):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        if audio_embs is not None:
            p_sep = "<SEP>"
            p_sep_tokens = self.llama_tokenizer(p_sep,
                return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        if self.args['use_lora'] == 'true':
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
            if audio_embs is not None:
                p_sep_embeds = self.llama_model.model.model.embed_tokens(p_sep_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s3 x embed_dim
        else:
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
            if audio_embs is not None:
                p_sep_embeds = self.llama_model.model.embed_tokens(p_sep_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s3 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        if self.args['use_lora'] == 'true':
            bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        else:
            bos_embeds = self.llama_model.model.embed_tokens(bos) # bsz x 1 x embed_dim

        if self.early_align and self.alignmode == 3:
            joint_embeds = torch.cat(audio_embs + [img_embeds], dim=1)
            inputs_embeds = torch.cat(
                [bos_embeds, p_before_embeds, joint_embeds, p_after_embeds], dim=1)
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size(1)+joint_embeds.size(1)],
                        dtype=torch.long).to(self.device).fill_(-100)
            )
            if audiomasks is not None:
                audiomasks = audiomasks if self.training else audiomasks * 0 + 1
                joint_masks = torch.cat([audiomasks, audiomasks.new_ones(batch_size, 1)], dim=-1).to(self.device)
                joint_masks = joint_masks.unsqueeze(-1).repeat(1, 1, self.num_video_query_token).view(batch_size, -1)
                atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size(1)], dtype=torch.long).to(self.device)
                atts_prefix = torch.cat([atts_prefix, joint_masks], dim=1)
            else:
                atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size(1)+joint_embeds.size(1)],
                    dtype=torch.long).to(self.device)
        elif audio_embs is not None:
            inputs_embeds = torch.cat(
                [bos_embeds, p_before_embeds, img_embeds, p_sep_embeds, audio_embs, p_after_embeds], dim=1)
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size(1)+img_embeds.size(1)+p_sep_embeds.size(1)+audio_embs.size(1)],
                        dtype=torch.long).to(self.device).fill_(-100)
            )
            if audiomasks is not None:
                audiomasks = audiomasks if self.training else audiomasks * 0 + 1
                visual_masks = audiomasks[:, 0].unsqueeze(1).repeat(1, img_embeds.size(1)).to(self.device)
                audio_masks = audiomasks[:, 1].unsqueeze(1).repeat(1, audio_embs.size(1)).to(self.device)
                atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size(1)],
                    dtype=torch.long).to(self.device)
                sep_masks = torch.ones([batch_size, p_sep_embeds.size(1)], dtype=torch.long).to(self.device)
                atts_prefix = torch.cat([atts_prefix, visual_masks, sep_masks, audio_masks], dim=1)
            else:
                atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size(1)+img_embeds.size(1)+p_sep_embeds.size(1)+audio_embs.size(1)],
                    dtype=torch.long).to(self.device)
        else:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1)
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+img_embeds.size(1)],
                        dtype=torch.long).to(self.device).fill_(-100)  
            )
            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+img_embeds.size(1)], dtype=torch.long).to(self.device)
        targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
        modality_lengths = atts_prefix.size(1)
        assert attention_mask.size() == targets.size() # bsz x (1 + s1 + S + s2)
        return inputs_embeds, targets, attention_mask, modality_lengths

    def calc_diversity_loss(self, query):
        dotprod = torch.einsum("bij,bkj->bik", query, query)
        modulus = torch.sqrt((query**2).sum(dim=-1))
        modulus = torch.einsum("bij,bjk->bik", modulus.unsqueeze(-1), modulus.unsqueeze(1))
        cos_sim = dotprod / (modulus + 1e-9)
        diag_mask = (1 - torch.eye(modulus.size(1), device=self.device)).unsqueeze(0)
        ave_sim = (cos_sim * diag_mask).sum() / ((modulus.size(1)**2 - modulus.size(1)) * modulus.size(0))
        return ave_sim

    def forward(self, inputs, reduction=True):
        # print("begin forward, {}".format(int(os.getenv('RANK', '0'))))

        output_texts = inputs['output_texts']
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer,
            output_texts,
            self.max_tgt_len,
            modality=inputs['modality'],
        )

        instruction_ids = None
        dummy_instruct = None
        # instruction_embs = None
        instructs = [text[0]["value"] for text in output_texts]
        if self.instructblip:
            instruction_ids = self.bert_tokenizer(
                instructs,
                padding='longest',
                truncation=True,
                max_length=self.max_tgt_len,
                return_tensors="pt",
            ).to(self.device)
            dummy_instruct = self.bert_tokenizer(["dummy"], return_tensors="pt").to(self.device)

        image_paths = inputs['image_paths']
        audio_embeds = None
        diversity_loss = 0
        # print(inputs['modality'])
        if inputs['modality'] == 'image':
            img_embeds, img_query = self.encode_image(image_paths, instruction_ids)
            if self.use_whisper:
                dummy, _ = self.encode_audio(dummy_audio_path, dummy_instruct)
                img_embeds = img_embeds + dummy.sum() * 0
            if self.diversity:
                diversity_loss += self.calc_diversity_loss(img_query)
        elif inputs['modality'] == 'video':
            img_embeds, _, video_query = self.encode_video(image_paths, instruction_ids)
            if self.diversity:
                diversity_loss += self.calc_diversity_loss(video_query)
        elif inputs['modality'] == 'audio':
            img_embeds, audio_query = self.encode_audio(image_paths, instruction_ids, earlyalign=self.early_align)
            if self.speech_qformer:
                dummy, _ = self.encode_image(dummy_image_path, dummy_instruct)
                img_embeds = img_embeds + dummy.sum() * 0
            if self.early_align:
                Tvideo = img_embeds.size(1) // 100 * 32 # // 25 * 32
                video_embs = img_embeds.new_zeros(img_embeds.size(0), Tvideo, img_embeds.size(2))
                video_masks = torch.ones(video_embs.size()[:-1], dtype=torch.long).to(video_embs.device)
                audiomasks = torch.tensor([0, 1]).unsqueeze(0).repeat(video_masks.size(0), 1).to(video_embs.device)
                img_embeds, _, audio_query = self.sequence_align_v2(video_embs, video_masks, img_embeds, audiomasks)
        elif inputs['modality'] == 'audioimage':
            image_paths = list(zip(*image_paths))
            query_mask = 1
            audio_embeds, audio_query = self.encode_audio(image_paths[0], instruction_ids, earlyalign=self.early_align)
            img_embeds, img_query = self.encode_image(image_paths[1], instruction_ids, earlyalign=self.early_align)
            inputs["audiomasks"] = None
            if self.early_align:
                Tvideo = audio_embeds.size(1) // 25  # 0.5s per frame
                video_embs = img_embeds.unsqueeze(1).repeat(1, Tvideo, 1, 1).view(img_embeds.size(0), -1, img_embeds.size(2))
                video_masks = torch.ones(video_embs.size()[:-1], dtype=torch.long).to(video_embs.device)
                audiomasks = torch.tensor([1, 1]).unsqueeze(0).repeat(video_masks.size(0), 1).to(video_embs.device)
                if self.alignmode == 1:
                    img_embeds, _, joint_query = self.sequence_align(
                        video_embs, video_masks, audio_embeds, audiomasks, instruction_inds=instruction_ids)
                else:
                    img_embeds, _, joint_query = self.sequence_align_v2(
                        video_embs, video_masks, audio_embeds, audiomasks, instruction_inds=instruction_ids)
                if self.alignmode == 3:
                    audio_embeds = [audio_query, img_query]
                else:
                    audio_embeds = None
            if self.diversity:
                if self.early_align:
                    diversity_loss = self.calc_diversity_loss(joint_query)
                else:
                    diversity_loss = self.calc_diversity_loss(img_query)
        elif inputs['modality'] == 'audiovideoimage':
            image_paths = list(zip(*image_paths))
            if self.cascaded == "audiogrounding":
                audio_embeds, audio_query = self.encode_audio(image_paths[0], instruction_ids, earlyalign=self.early_align)
                img_embeds, img_mask, video_query = self.encode_video(
                    image_paths[1], instruction_ids, earlyalign=self.early_align, audio_query=audio_query)
            elif self.cascaded == "bothgrounding":
                _, pre_audio_query = self.encode_audio(image_paths[0], instruction_ids, earlyalign=self.early_align)
                _, _, pre_video_query = self.encode_video(image_paths[1], instruction_ids, earlyalign=self.early_align)
                audio_embeds, audio_query = self.encode_audio(
                    image_paths[0], instruction_ids, earlyalign=self.early_align, visual_query=pre_video_query)
                img_embeds, img_mask, video_query = self.encode_video(
                    image_paths[1], instruction_ids, earlyalign=self.early_align, audio_query=pre_audio_query)
            else:
                audio_embeds, audio_query = self.encode_audio(image_paths[0], instruction_ids, earlyalign=self.early_align)
                img_embeds, img_mask, video_query = self.encode_video(
                    image_paths[1], instruction_ids, earlyalign=self.early_align)
            if self.early_align:
                if self.alignmode == 1:
                    img_embeds, _, joint_query = self.sequence_align(
                        img_embeds, img_mask, audio_embeds, inputs["audiomasks"].to(self.device), instruction_inds=instruction_ids)
                else:
                    img_embeds, _, joint_query = self.sequence_align_v2(
                        img_embeds, img_mask, audio_embeds, inputs["audiomasks"].to(self.device), instruction_inds=instruction_ids)
                if self.alignmode == 3:
                    audio_embeds = [audio_query, video_query]
                else:
                    audio_embeds = None
            if self.diversity:
                if self.early_align:
                    diversity_loss = self.calc_diversity_loss(joint_query)
                else:
                    diversity_loss = self.calc_diversity_loss(video_query)
        else:
            raise Exception("Undefined modality type")
        # print("Finished encoder, {}".format(int(os.getenv('RANK', '0'))))

        inputs_embeds, targets, attention_mask, modality_lengths = self.prompt_wrap(
            img_embeds,
            input_ids,
            target_ids,
            attention_mask,
            audio_embs=audio_embeds,
            audiomasks=inputs["audiomasks"],
        )
        # print("Finished prompt wrap, {}".format(int(os.getenv('RANK', '0'))))

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            modality_lengths=modality_lengths if self.modalitymask else 0,
        )
        loss = outputs.loss
        if self.diversity:
            if not self.training:
                print(diversity_loss)
            loss += diversity_loss * self.diversity_loss_factor
        # print("Finished vicuna forward, {}".format(int(os.getenv('RANK', '0'))))
        # calculate the token accuarcy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask    # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    def extract_multimodal_feature(self, inputs):
        features = []
        instruction_ids = None
        instruction_embs = None
        instructs = [inputs["prompt"]]
        if self.instructblip:
            instruction_ids = self.bert_tokenizer(
                instructs,
                padding='longest',
                truncation=True,
                max_length=self.max_tgt_len,
                return_tensors="pt",
            ).to(self.device)
        
        feature_dict = {}
        audio_query = None
        img_query = None
        video_query = None
        if inputs['image_paths']:
            if self.cascaded in ["audiogrounding", "bothgrounding"] and inputs['audio_paths']:
                _, audio_query = self.encode_audio(inputs['audio_paths'], instruction_ids)
            elif self.cascaded == ["audiogrounding", "bothgrounding"]:
                _, audio_query = self.encode_audio(dummy_audio_path, instruction_ids)
                audio_query = audio_query * 0
            image_embeds, image_query = self.encode_image(inputs['image_paths'], instruction_ids, audio_query=audio_query, earlyalign=self.early_align)
            feature_dict["image"] = image_embeds
        if inputs['audio_paths']:
            pre_vis_query = None
            if self.cascaded == "bothgrounding" and inputs['image_paths']:
                _, pre_vis_query = self.encode_image(inputs['image_paths'], instruction_ids)
            elif self.cascaded == "bothgrounding" and inputs['video_paths']:
                _, _, pre_vis_query = self.encode_video(inputs['video_paths'], instruction_ids)
            elif self.cascaded == "bothgrounding":
                _, pre_vis_query = self.encode_image(dummy_image_path, instruction_ids)
                pre_vis_query = pre_vis_query * 0
            audio_embeds, audio_query = self.encode_audio(inputs['audio_paths'], instruction_ids, visual_query=pre_vis_query, earlyalign=self.early_align)
            feature_dict["audio"] = audio_embeds
        if inputs['video_paths']:
            pre_audio_query = None
            if self.cascaded in ["audiogrounding", "bothgrounding"] and inputs['audio_paths']:
                _, pre_audio_query = self.encode_audio(inputs['audio_paths'], instruction_ids)
            elif self.cascaded in ["audiogrounding", "bothgrounding"]:
                _, pre_audio_query = self.encode_audio(dummy_audio_path, instruction_ids)
                pre_audio_query = pre_audio_query * 0
            video_embeds, video_mask, video_query = self.encode_video(inputs['video_paths'], instruction_ids, earlyalign=self.early_align, audio_query=pre_audio_query)
            feature_dict["video"] = video_embeds

        if self.early_align:
            if not inputs['audio_paths'] and inputs["video_paths"]:
                audio_embeds = video_embeds.new_zeros(
                    video_embeds.size(0),
                    video_embeds.size(1)//32*25,
                    self.speech_encoder.config.d_model if self.speech_qformer else video_embeds.size(-1),
                )
                audiomasks = torch.tensor([1, 0]).unsqueeze(0).repeat(video_mask.size(0), 1).to(video_embeds.device)
            elif not inputs["video_paths"]:
                if inputs["image_paths"]:
                    if not inputs["audio_paths"]:
                        inputs["audio_paths"] = dummy_audio_path
                        audio_embeds, audio_query = self.encode_audio(inputs['audio_paths'], instruction_ids, earlyalign=self.early_align)
                        audio_embeds = audio_embeds * 0
                    if inputs["audio_paths"]:
                        video_embeds = image_embeds.unsqueeze(1).repeat(1, audio_embeds.size(1)//25, 1, 1).view(
                            image_embeds.size(0), -1, image_embeds.size(2))
                        video_mask = audio_embeds.new_ones(video_embeds.size()[:-1])
                        audiomasks = torch.tensor([1, 1]).unsqueeze(0).repeat(video_mask.size(0), 1).to(self.device)
                    else:
                        video_embeds = image_embeds.unsqueeze(1).repeat(1, 60, 1, 1).view(
                            image_embeds.size(0), -1, image_embeds.size(2))
                        audio_embeds = video_embeds.new_zeros(
                            video_embeds.size(0),
                            video_embeds.size(1)//32*25,
                            self.speech_encoder.config.d_model if self.speech_qformer else video_embeds.size(-1),
                        )
                        video_mask = audio_embeds.new_ones(video_embeds.size()[:-1])
                        audiomasks = torch.tensor([1, 0]).unsqueeze(0).repeat(video_mask.size(0), 1).to(video_embeds.device)
                elif inputs["audio_paths"]:
                    video_embeds = audio_embeds.new_zeros(
                        audio_embeds.size(0),
                        audio_embeds.size(1)//25*32,
                        self.visual_hidden_size if self.speech_qformer else audio_embeds.size(2),
                    ).to(self.device)
                    # image_embeds, pre_vis_query = self.encode_image(dummy_image_path, instruction_ids, earlyalign=self.early_align)
                    # video_embeds = image_embeds.unsqueeze(1).repeat(1, audio_embeds.size(1)//25, 1, 1).view(
                    #     image_embeds.size(0), -1, image_embeds.size(2))
                    video_mask = audio_embeds.new_ones(video_embeds.size()[:-1]).to(self.device)
                    audiomasks = torch.tensor([1, 1]).unsqueeze(0).repeat(video_mask.size(0), 1).to(self.device)
                else:
                    raise Exception("Early align mode has to have either audio or video!")
            elif inputs["video_paths"] and inputs["audio_paths"]:
                audiomasks = torch.tensor([1, 1]).unsqueeze(0).repeat(video_mask.size(0), 1).to(self.device)
            else:
                raise Exception("Early align mode has to have either audio or video!")
            if self.alignmode == 1:
                video_embeds, _, video_query = self.sequence_align(
                    video_embeds, video_mask, audio_embeds, audiomasks, instruction_inds=instruction_ids)
            else:
                video_embeds, _, video_query = self.sequence_align_v2(
                    video_embeds, video_mask, audio_embeds, audiomasks, instruction_inds=instruction_ids)
            if self.diversity:
                # Calculate cosine similarity
                ave_sim = self.calc_diversity_loss(video_query)
                post_ave_sim = self.calc_diversity_loss(video_embeds)
                print("Average cosine similarity: {}".format(ave_sim))
                print("Average post cosine similarity: {}".format(post_ave_sim))
            if self.alignmode == 3:
                video_embeds = torch.cat([audio_query, video_query if video_query is not None else image_query, video_embeds], dim=1)
            feature_dict["video"] = video_embeds
            feature_dict["qformerout"] = video_query
        features = []
        if "video" in feature_dict:
            features.append(feature_dict["video"])
        elif "image" in feature_dict:
            features.append(feature_dict["image"])
        if "audio" in feature_dict and not self.early_align:
            features.append(feature_dict["audio"])
        return features, feature_dict

    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        if len(inputs['modality_embeds']) == 1:
            feature_embeds, feature_dict = inputs['modality_embeds'][0]
        else:
            feature_embeds, feature_dict = self.extract_multimodal_feature(inputs)
            inputs['modality_embeds'].append(feature_embeds)
        modality_mask = inputs["avmask"] if "avmask" in inputs else [1, 1]

        batch_size = feature_embeds[0].shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        if self.args['use_lora'] == 'true':
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        else:
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        text = '</Img> ' + prompt + '\nASSISTANT:'
        textsep = '<SEP>'
        sep_tokens = self.llama_tokenizer(textsep, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        if self.args['use_lora'] == 'true':
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
            sep_embeds = self.llama_model.model.model.embed_tokens(sep_tokens.input_ids).expand(batch_size, -1, -1)
        else:
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
            sep_embeds = self.llama_model.model.embed_tokens(sep_tokens.input_ids).expand(batch_size, -1, -1)
            # delete later
            # p_after_embeds = self.llama_tokenizer("</s>" + prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            # p_after_embeds = self.llama_model.model.embed_tokens(p_after_embeds.input_ids).expand(batch_size, -1, -1)
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        if self.args['use_lora'] == 'true':
            bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        else:
            bos_embeds = self.llama_model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        if len(feature_embeds) == 1:
            feature_embeds[0] = feature_embeds[0] * modality_mask[0]
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds[0], p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # delete later
            # inputs_embeds = torch.cat([feature_embeds[0], p_after_embeds], dim=1)
        else:
            totalemb = [bos_embeds, p_before_embeds]
            for k, feature_emb in enumerate(feature_embeds[:-1]):
                totalemb.append(feature_emb * modality_mask[k])
                totalemb.append(sep_embeds)
            totalemb.append(feature_embeds[-1] * modality_mask[-1])
            totalemb.append(p_after_embeds)
            inputs_embeds = torch.cat(totalemb, dim=1)
        return inputs_embeds, feature_dict

    def generate(self, inputs):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''
        input_embeds, feature_dict = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2], encounters=1)])
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            min_length=1,
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=inputs.get('dosample', False),
            use_cache=True,
            num_beams=inputs.get('num_beams', 1),
            repetition_penalty=1.5,
            length_penalty=inputs.get('lengthpenalty', 1),
            stopping_criteria=stopping_criteria,
            num_return_sequences=inputs.get('num_returns', 1),
        )
        if inputs.get('num_returns', 1) > 1:
            output_text = [self.llama_tokenizer.decode(output, skip_special_tokens=True).split("\n#")[0] for output in outputs]
        else:
            output_text = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "\n#" in output_text:
                output_text = output_text.split("\n#")[0]
        return output_text

    def calc_entropy(self, inputs):
        with torch.no_grad():
            input_embeds = self.extract_multimodal_feature(inputs)
            output_texts = inputs["prompt"]
            input_ids, target_ids, attention_mask = process_batch_instance(
                self.llama_tokenizer,
                output_texts,
                self.max_tgt_len,
            )
            inputs_embeds, targets, attention_mask = self.prompt_wrap(
                input_embeds[0],
                input_ids,
                target_ids,
                attention_mask,
            )
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            logits = outputs.logits[0][-(targets[0] != -100).sum().item():]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=-1)
        return entropy
