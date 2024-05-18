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

Download model checkpoint [here](https://drive.google.com/drive/folders/166g9WVWXwYP77VJyOd3isi_UvRpmX_cv?usp=sharing) and put the folder under `ckpt/`


## Inference
```
conda activate favor
python inference.py
==========================Output=============================
The video is a romantic scene of a man and a woman on a boat. The man is holding the woman in his arms, and they are both looking at the sunset. The audio is a song that adds to the romantic atmosphere. The woman says "I'm flying" and "Jack," which suggests that they are happy and enjoying the moment. The setting of the boat and the sunset create a beautiful and serene environment that enhances the romantic feel of the video. The man and the woman's body language and facial expressions also convey their love and affection for each other. Overall, the video is a perfect representation of a romantic and intimate moment between two people.
```

## Reference
Please cite our paper if you use our model
```
@article{sun2023finegrained,
      title={Fine-grained Audio-Visual Joint Representations for Multimodal Large Language Models}, 
      author={Guangzhi Sun and Wenyi Yu and Changli Tang and Xianzhao Chen and Tian Tan and Wei Li and Lu Lu and Zejun Ma and Chao Zhang},
      year={2023},
      journal={arXiv:2310.05863},
}
```
