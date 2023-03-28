


# Text2Video-Zero

This repository is the official implementation of [Text2Video-Zero](https://arxiv.org/abs/2303.13439).


**[Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://arxiv.org/abs/2303.13439)**
</br>
Levon Khachatryan,
Andranik Movsisyan,
Vahram Tadevosyan,
Roberto Henschel,
[Zhangyang Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang), Shant Navasardyan, [Humphrey Shi](https://www.humphreyshi.com)
</br>

[Paper](https://arxiv.org/abs/2303.13439) | [Video](https://www.dropbox.com/s/uv90mi2z598olsq/Text2Video-Zero.MP4?dl=0) | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PAIR/Text2Video-Zero) 


<p align="center">
<img src="__assets__/github/teaser/teaser_final.png" width="800px"/>  
<br>
<em>Our method Text2Video-Zero enables zero-shot video generation using (i) a textual prompt (see rows 1, 2),  (ii) a prompt combined with guidance from poses or edges (see lower right), and  (iii)  Video Instruct-Pix2Pix, i.e., instruction-guided video editing (see lower left). 
    Results are temporally consistent and follow closely the guidance and textual prompts.</em>
</p>

## News

* [03/23/2023] Paper [Text2Video-Zero](https://arxiv.org/abs/2303.13439) released!
* [03/25/2023] The [first version](https://huggingface.co/spaces/PAIR/Text2Video-Zero) of our huggingface demo (containing `zero-shot text-to-video generation` and  `Video Instruct Pix2Pix`) released!
* [03/27/2023] The [full version](https://huggingface.co/spaces/PAIR/Text2Video-Zero) of our huggingface demo released! Now also included: `text and pose conditional video generation`, `text and canny-edge conditional video generation`, and 
`text, canny-edge and dreambooth conditional video generation`.
* [03/28/2023] Code for all our generation methods released! We added a new low-memory setup. Minimum required GPU VRAM is currently **12 GB**. It will be further reduced in the upcoming releases. 

## Setup




### Requirements

```shell
pip install -r requirements.txt
```
<!--- Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. --->

### Weights

#### Text-To-Video with Pose Guidance

Download the pose model weights used in [ControlNet](https://arxiv.org/abs/2302.05543):
```shell
wget -P annotator/ckpts https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth
wget -P annotator/ckpts https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth
```


<!---
#### Text-To-Video
Any [Stable Diffusion](https://arxiv.org/abs/2112.10752) v1.4 model weights in huggingface format can be used and must be placed in `models/text-to-video`.
For instance:

```shell
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 model_weights
mv model_weights models/text-to-video
```

#### Video Instruct-Pix2Pix
From [Instruct-Pix2Pix](https://arxiv.org/pdf/2211.09800.pdf) download pretrained model files:
```shell
git lfs install
git clone https://huggingface.co/timbrooks/instruct-pix2pix models/instruct-pix2pix
``` 

#### Text-To-Video with Pose Guidance
From [ControlNet](https://arxiv.org/abs/2302.05543), download the open pose model file:
```shell
mkdir -p models/control
wget -P models/control https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth
```
#### Text-To-Video with Edge Guidance
From [ControlNet](https://arxiv.org/abs/2302.05543), download the Canny edge model file:
```shell
mkdir -p models/control
wget -P models/control https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth 
```

--->


#### Text-To-Video with Edge Guidance and Dreambooth

Integrate a `SD1.4` Dreambooth model into ControlNet using [this](https://github.com/lllyasviel/ControlNet/discussions/12) procedure. Load the model into `models/control_db/`. Dreambooth models can be obtained, for instance, from [CIVITAI](https://civitai.com). 

We provide already prepared model files for `anime` (keyword `1girl`), `arcane style` (keyword `arcane style`) `avatar` (keyword `avatar style`) and `gta-5 style`  (keyword `gtav style`). To this end, download the model files from [google drive](https://drive.google.com/drive/folders/1uwXNjJ-4Ws6pqyjrIWyVPWu_u4aJrqt8?usp=share_link) and extract them into `models/control_db/`.

## Inference API

To run inferences create an instance of `Model` class
```python
import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)
```

---


### Text-To-Video
To directly call our text-to-video generator, run this python command which stores the result in `tmp/text2video/A_horse_galloping_on_a_street.mp4` :
```python
from pathlib import Path

prompt = "A horse galloping on a street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

out_path, fps = Path(f"tmp/text2video/{prompt.replace(' ','_')}.mp4"), 4
if not out_path.parent.exists():
  out_path.parent.mkdir(parents=True)
model.process_text2video(prompt, fps = fps, path = out_path.as_posix(), **params)
```

#### Hyperparameters (Optional)

You can define the following hyperparameters:
* **Motion field strength**:   `motion_field_strength_x` = $\delta_x$  and `motion_field_strength_y` = $\delta_x$ (see our paper, Sect. 3.3.1). Default: `motion_field_strength_x=motion_field_strength_y= 12`.
* $T$ and $T'$ (see our paper, Sect. 3.3.1). Define values `t0` and `t1` in the range `{0,...,50}`. Default: `t0=44`, `t1=47` (DDIM steps). Corresponds to timesteps `881` and `941`, respectively. 
* **Video length**: Define the number of frames `video_length` to be generated. Default: `video_length=8`.


---

### Text-To-Video with Pose Control
To directly call our text-to-video generator with pose control, run this python command:
```python
from pathlib import Path

prompt = 'an astronaut dancing in outer space'
motion_path = Path('__assets__/poses_skeleton_gifs/dance1_corr.mp4')
out_path = Path(f'./{prompt}.gif')
model.process_controlnet_pose(motion_path.as_posix(), prompt=prompt, save_path=out_path.as_posix())
```


---

### Text-To-Video with Edge Control
To directly call our text-to-video generator with edge control, run this python command:
```python
from pathlib import Path

prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
video_path = Path('__assets__/canny_videos_mp4/deer.mp4')
out_path = Path(f'./{prompt}.mp4')
model.process_controlnet_canny(video_path.as_posix(), prompt=prompt, save_path=out_path.as_posix())
```

#### Hyperparameters

You can define the following hyperparameters for Canny edge detection:
* **low threshold**. Define value `low_threshold` in the range $(0, 255)$. Default: `low_threshold=100`.
* **high threshold**. Define value `high_threshold` in the range $(0, 255)$. Default: `high_threshold=200`. Make sure that `high_threshold` > `low_threshold`.

You can give hyperparameters as arguments to `model.process_controlnet_canny`

---


### Text-To-Video with Edge Guidance and Dreambooth specialization
Load a dreambooth model then proceed as described in `Text-To-Video with Edge Guidance`
```python
from pathlib import Path

prompt = 'your prompt'
video_path = Path('path/to/your/video')
dreambooth_model_path = Path('path/to/your/dreambooth/model')
out_path = Path(f'./{prompt}.gif')
model.process_controlnet_canny_db(dreambooth_model_path.as_posix(), video_path.as_posix(), prompt=prompt, save_path=out_path.as_posix())
```
---

### Video Instruct-Pix2Pix

To perform pix2pix video editing, run this python command:
```python
from pathlib import Path

prompt = 'make it Van Gogh Starry Night'
video_path = Path('__assets__/pix2pix video/camel.mp4')
out_path = Path(f'./{prompt}.mp4')
model.process_pix2pix(video_path.as_posix(), prompt=prompt, save_path=out_path.as_posix())
```

---

### Low Memory Inference
Each of the above introduced interface can be run in a low memory setup. In the minimal setup, a GPU with **12 GB VRAM** is sufficient. 

To reduce the memory usage, add `chunk_size=k` as additional parameter when calling one of the above defined inference APIs. The integer value `k` must be in the range `{2,...,video_length}`. It defines the number of frames that are processed at once (without any loss in quality). The lower the value the less memory is needed.


We plan to release soon a new version that further reduces the memory usage. 


---


### Ablation Study
To replicate the ablation study, add additional parameters when calling the above defined inference APIs.
*  To deactivate `cross-frame attention`: Add `use_cf_attn=False` to the parameter list.
* To deactivate enriching latent codes with `motion dynamics`: Add `use_motion_field=False` to the parameter list.

---

## Inference using Gradio
From the project root folder, run this shell command:
```shell
python app.py
```

Then access the app [locally](http://127.0.0.1:7860) with a browser.



## Results

### Text-To-Video
<table class="center">
<tr>
  <td><img src="__assets__/github/results/t2v/cat_running.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/playing.gif"></td>
  <td><img src="__assets__/github/results/t2v/running.gif"></td>              
  <td><img src="__assets__/github/results/t2v/skii.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A cat is running on the grass"</td>
  <td width=25% style="text-align:center;">"A panda is playing guitar on times square</td>
  <td width=25% style="text-align:center;">"A man is running in the snow"</td>
  <td width=25% style="text-align:center;">"An astronaut is skiing down the hill"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/t2v/panda_surfing.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/bear_dancing.gif"></td>
  <td><img src="__assets__/github/results/t2v/bicycle.gif"></td>              
  <td><img src="__assets__/github/results/t2v/horse_galloping.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A panda surfing on a wakeboard"</td>
  <td width=25% style="text-align:center;">"A bear dancing on times square</td>
  <td width=25% style="text-align:center;">"A man is riding a bicycle in the sunshine"</td>
  <td width=25% style="text-align:center;">"A horse galloping on a street"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/t2v/tiger_walking.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/panda_surfing_2.gif"></td>
  <td><img src="__assets__/github/results/t2v/horse_galloping_2.gif"></td>              
  <td><img src="__assets__/github/results/t2v/cat_walking.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A tiger walking alone down the street"</td>
  <td width=25% style="text-align:center;">"A panda surfing on a wakeboard</td>
  <td width=25% style="text-align:center;">"A horse galloping on a street"</td>
  <td width=25% style="text-align:center;">"A cute cat running in a beatiful meadow"</td>
</tr>


<tr>
  <td><img src="__assets__/github/results/t2v/horse_galloping_3.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/panda_walking.gif"></td>
  <td><img src="__assets__/github/results/t2v/dog_walking.gif"></td>              
  <td><img src="__assets__/github/results/t2v/astronaut.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A horse galloping on a street"</td>
  <td width=25% style="text-align:center;">"A panda walking alone down the street</td>
  <td width=25% style="text-align:center;">"A dog is walking down the street"</td>
  <td width=25% style="text-align:center;">"An astronaut is waving his hands on the moon"</td>
</tr>


</table>

### Text-To-Video with Pose Guidance


<table class="center">
<tr>
  <td><img src="__assets__/github/results/pose2v/img_bot_left.gif" raw=true><img src="__assets__/github/results/pose2v/pose_bot_left.gif"></td>
  <td><img src="__assets__/github/results/pose2v/img_bot_right.gif" raw=true><img src="__assets__/github/results/pose2v/pose_bot_right.gif"></td>
  <td><img src="__assets__/github/results/pose2v/img_top_left.gif" raw=true><img src="__assets__/github/results/pose2v/pose_top_left.gif"></td>
  <td><img src="__assets__/github/results/pose2v/img_top_right.gif" raw=true><img src="__assets__/github/results/pose2v/pose_top_right.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A bear dancing on the concrete"</td>
  <td width=25% style="text-align:center;">"An alien dancing under a flying saucer</td>
  <td width=25% style="text-align:center;">"A panda dancing in Antarctica"</td>
  <td width=25% style="text-align:center;">"An astronaut dancing in the outer space"</td>

</tr>
</table>

### Text-To-Video with Edge Guidance



<table class="center">
<tr>
  <td><img src="__assets__/github/results/edge2v/butterfly.gif" raw=true><img src="__assets__/github/results/edge2v/butterfly_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/head.gif" raw=true><img src="__assets__/github/results/edge2v/head_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/jelly.gif" raw=true><img src="__assets__/github/results/edge2v/jelly_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/mask.gif" raw=true><img src="__assets__/github/results/edge2v/mask_edge.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"White butterfly"</td>
  <td width=25% style="text-align:center;">"Beautiful girl</td>
    <td width=25% style="text-align:center;">"A jellyfish"</td>
  <td width=25% style="text-align:center;">"beautiful girl halloween style"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/edge2v/fox.gif" raw=true><img src="__assets__/github/results/edge2v/fix_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/head_2.gif" raw=true><img src="__assets__/github/results/edge2v/head_2_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/santa.gif" raw=true><img src="__assets__/github/results/edge2v/santa_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/dear.gif" raw=true><img src="__assets__/github/results/edge2v/dear_edge.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"Wild fox is walking"</td>
  <td width=25% style="text-align:center;">"Oil painting of a beautiful girl close-up</td>
    <td width=25% style="text-align:center;">"A santa claus"</td>
  <td width=25% style="text-align:center;">"A deer"</td>
</tr>

</table>


### Text-To-Video with Edge Guidance and Dreambooth specialization




<table class="center">
<tr>
  <td><img src="__assets__/github/results/canny_db/anime_style.gif" raw=true><img src="__assets__/github/results/canny_db/anime_edge.gif"></td>
  <td><img src="__assets__/github/results/canny_db/arcane_style.gif" raw=true><img src="__assets__/github/results/canny_db/arcane_edge.gif"></td>
  <td><img src="__assets__/github/results/canny_db/gta-5_man_style.gif" raw=true><img src="__assets__/github/results/canny_db/gta-5_man_edge.gif"></td>
  <td><img src="__assets__/github/results/canny_db/img_bot_right.gif" raw=true><img src="__assets__/github/results/canny_db/edge_bot_right.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"anime style"</td>
  <td width=25% style="text-align:center;">"arcane style</td>
    <td width=25% style="text-align:center;">"gta-5 man"</td>
  <td width=25% style="text-align:center;">"avatar style"</td>
</tr>

</table>


## Video Instruct Pix2Pix

<table class="center">
<tr>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/up_left.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/bot_left.gif"></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/up_mid.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/bot_mid.gif"></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/up_right.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/bot_right.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"Replace man with chimpanze"</td>
  <td width=25% style="text-align:center;">"Make it Van Gogh Starry Night style"</td>
    <td width=25% style="text-align:center;">"Make it Picasso style"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/up_left.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/bot_left.gif"></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/up_mid.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/bot_mid.gif"></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/up_right.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/bot_right.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"Make it Expressionism style"</td>
  <td width=25% style="text-align:center;">"Make it night"</td>
    <td width=25% style="text-align:center;">"Make it autumn"</td>
</tr>
</table>

## License
Our code is published under the CreativeML Open RAIL-M license. The license provided in this repository applies to all additions and contributions we make upon the original stable diffusion code. The original stable diffusion code is under the CreativeML Open RAIL-M license, which can found [here](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).


## BibTeX
If you use our work in your research, please cite our publication:
```
@article{text2video-zero,
    title={Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators},
    author={Khachatryan, Levon and Movsisyan, Andranik and Tadevosyan, Vahram and Henschel, Roberto and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
    journal={arXiv preprint arXiv:2303.13439},
    year={2023}
}
```
