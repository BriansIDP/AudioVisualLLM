# FAVOR
Fine-grained Audio-Visual Joint Representations for Multimodal Large Language Models

<a href='https://881c5a6a6db84b1a2f.gradio.live'><img src='https://img.shields.io/badge/gradio-demo-blue'></a>

Button Specifications:

`Clear All`: clear chat history as well as all modality inputs. **Please always use clear all before you want to upload or update any image, audio or video** 

`Clear history`: only clear chat history. The modality input will remain unchanged unless you click `Clear All`.

`Submit`: submit the text in the text box to get a response

`Resubmit`: clear the previous conversation turn and then submit the text in the text box

`maximum length`, `top p` and `temperature` have their meanings

Examples mentioned in the paper are provided. Please feel free to start with those.


## Getting started
```
cd AudioVisualLLM
conda env create -f environment.yml
mkdir ckpt
```

Download model checkpoint [here]([https://duckduckgo.com](https://drive.google.com/drive/folders/166g9WVWXwYP77VJyOd3isi_UvRpmX_cv?usp=sharing)) and put the folder under `ckpt/`


## Inference
```
conda activate favor
python inference.py
==========================Output=============================

```
