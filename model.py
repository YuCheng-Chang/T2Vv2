from enum import Enum
import gc
import numpy as np
import tomesd
import torch

from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler
from text_to_video_pipeline import TextToVideoPipeline

import utils
import gradio_utils
import os
on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"

<<<<<<< HEAD

=======
#紀鵬到此一遊:)
>>>>>>> 0d598340d0636e7a3ff895ad74b8444225852e19
class ModelType(Enum):
    Pix2Pix_Video = 1,
    Text2Video = 2,
    ControlNetCanny = 3,
    ControlNetCannyDB = 4,
    ControlNetPose = 5,
    ControlNetDepth = 6,


class Model:
    def __init__(self, device, dtype, **kwargs):
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        self.pipe_dict = {
            ModelType.Pix2Pix_Video: StableDiffusionInstructPix2PixPipeline,
            ModelType.Text2Video: TextToVideoPipeline,
            ModelType.ControlNetCanny: StableDiffusionControlNetPipeline,
            ModelType.ControlNetCannyDB: StableDiffusionControlNetPipeline,
            ModelType.ControlNetPose: StableDiffusionControlNetPipeline,
            ModelType.ControlNetDepth: StableDiffusionControlNetPipeline,
        }
        self.controlnet_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=2)
        self.pix2pix_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=3)
        self.text2video_attn_proc = utils.CrossFrameAttnProcessor(
            unet_chunk_size=2)

        self.pipe = None
        self.model_type = None

        self.states = {}
        self.model_name = ""

    def set_model(self, model_type: ModelType, model_id: str, **kwargs):
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        safety_checker = kwargs.pop('safety_checker', None)
        self.pipe = self.pipe_dict[model_type].from_pretrained(
            model_id, safety_checker=safety_checker, **kwargs).to(self.device).to(self.dtype)
        self.model_type = model_type
        self.model_name = model_id

    def inference_chunk(self, frame_ids, **kwargs):
        if not hasattr(self, "pipe") or self.pipe is None:
<<<<<<< HEAD
            return

        prompt = np.array(kwargs.pop('prompt'))
        negative_prompt = np.array(kwargs.pop('negative_prompt', ''))
        latents = None
        if 'latents' in kwargs:
            latents = kwargs.pop('latents')[frame_ids]
        if 'image' in kwargs:
            kwargs['image'] = kwargs['image'][frame_ids]
        if 'video_length' in kwargs:
            kwargs['video_length'] = len(frame_ids)
        if self.model_type == ModelType.Text2Video:
            kwargs["frame_ids"] = frame_ids
        return self.pipe(prompt=prompt[frame_ids].tolist(),
                         negative_prompt=negative_prompt[frame_ids].tolist(),
                         latents=latents,
                         generator=self.generator,
                         **kwargs)

    def inference(self, split_to_chunks=False, chunk_size=8, **kwargs):
=======
            print("Error: pipe is not initialized")
            return

        print(f"Processing chunk with frame_ids: {frame_ids}")

        def check_tensor(tensor, name):
            if tensor is None:
                print(f"{name} is None")
                return
            if torch.isnan(tensor).any():
                print(f"NaN found in {name}")
            if torch.isinf(tensor).any():
                print(f"Inf found in {name}")
            print(f"{name} shape: {tensor.shape}")
            print(f"{name} min: {tensor.min().item()}, max: {tensor.max().item()}")

        prompt = kwargs.pop('prompt', None)
        negative_prompt = kwargs.pop('negative_prompt', None)
        
        # print(f"Prompt: {prompt}")
        # print(f"Negative prompt: {negative_prompt}")

        latents = kwargs.pop('latents', None)
        if latents is not None:
            latents = latents[frame_ids]
            check_tensor(latents, "latents")
        else:
            print("latent is None")
            
        if 'image' in kwargs:
            kwargs['image'] = kwargs['image'][frame_ids]
            check_tensor(kwargs['image'], "image")

        if 'video_length' in kwargs:
            kwargs['video_length'] = len(frame_ids)

        if self.model_type == ModelType.Text2Video:
            kwargs["frame_ids"] = frame_ids

        try:
            # 檢查輸入
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    check_tensor(value, f"Input {key}")
            # self.pipe = self.pipe.float()
            result = self.pipe(
                prompt=prompt[frame_ids].tolist() if isinstance(prompt, np.ndarray) else prompt,
                negative_prompt=negative_prompt[frame_ids].tolist() if isinstance(negative_prompt, np.ndarray) else negative_prompt,
                latents=latents,
                generator=self.generator,
                **kwargs
            )
             # 檢查中間結果
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    check_tensor(value, f"Output {key}")
            check_tensor(result.images, "result.images")
            return result
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def inference(self, split_to_chunks=False, chunk_size=8, **kwargs):
        def print_cuda_memory():
            if torch.cuda.is_available():
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
>>>>>>> 0d598340d0636e7a3ff895ad74b8444225852e19
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        if "merging_ratio" in kwargs:
            merging_ratio = kwargs.pop("merging_ratio")

            # if merging_ratio > 0:
            tomesd.apply_patch(self.pipe, ratio=merging_ratio)
        seed = kwargs.pop('seed', 0)
        if seed < 0:
            seed = self.generator.seed()
        kwargs.pop('generator', '')

        if 'image' in kwargs:
            f = kwargs['image'].shape[0]
        else:
            f = kwargs['video_length']

        assert 'prompt' in kwargs
        prompt = [kwargs.pop('prompt')] * f
        negative_prompt = [kwargs.pop('negative_prompt', '')] * f

        frames_counter = 0

        # Processing chunk-by-chunk
        if split_to_chunks:
            chunk_ids = np.arange(0, f, chunk_size - 1)
            result = []
            for i in range(len(chunk_ids)):
<<<<<<< HEAD
                ch_start = chunk_ids[i]
                ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                frame_ids = [0] + list(range(ch_start, ch_end))
                self.generator.manual_seed(seed)
                print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
                result.append(self.inference_chunk(frame_ids=frame_ids,
                                                   prompt=prompt,
                                                   negative_prompt=negative_prompt,
                                                   **kwargs).images[1:])
=======
                print_cuda_memory()
                ch_start = chunk_ids[i]
                ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                frame_ids = [0] + list(range(ch_start, ch_end))
                print(f'Frame ids {frame_ids}')
                self.generator.manual_seed(seed)
                print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
                chunk_result = self.inference_chunk(frame_ids=frame_ids,
                                                    prompt=prompt,
                                                    negative_prompt=negative_prompt,
                                                    **kwargs).images[1:]
                
                # # 修改這些檢查以使用 PyTorch 函數
                # print(f"Chunk {i+1} shape: {chunk_result.shape}")
                # print(f"Chunk {i+1} min: {torch.min(chunk_result).item()}, max: {torch.max(chunk_result).item()}")
                # print(f"Chunk {i+1} has NaN: {torch.isnan(chunk_result).any().item()}")
                
                result.append(chunk_result)
>>>>>>> 0d598340d0636e7a3ff895ad74b8444225852e19
                frames_counter += len(chunk_ids)-1
                if on_huggingspace and frames_counter >= 80:
                    break
            result = np.concatenate(result)
            return result
        else:
            self.generator.manual_seed(seed)
            return self.pipe(prompt=prompt, negative_prompt=negative_prompt, generator=self.generator, **kwargs).images

    def process_controlnet_canny(self,
                                 video_path,
                                 prompt,
                                 chunk_size=8,
                                 watermark='Picsart AI Research',
                                 merging_ratio=0.0,
                                 num_inference_steps=20,
                                 controlnet_conditioning_scale=1.0,
                                 guidance_scale=9.0,
                                 seed=42,
                                 eta=0.0,
                                 low_threshold=100,
                                 high_threshold=200,
                                 resolution=512,
                                 use_cf_attn=True,
                                 save_path=None):
        print("Module Canny")
        video_path = gradio_utils.edge_path_to_video_path(video_path)
        if self.model_type != ModelType.ControlNetCanny:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny")
            self.set_model(ModelType.ControlNetCanny,
                           model_id="runwayml/stable-diffusion-v1-5", controlnet=controlnet)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.controlnet_attn_proc)
                self.pipe.controlnet.set_attn_processor(
                    processor=self.controlnet_attn_proc)

        added_prompt = 'best quality, extremely detailed'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False)
        control = utils.pre_process_canny(
            video, low_threshold, high_threshold).to(self.device).to(self.dtype)

        # canny_to_save = list(rearrange(control, 'f c w h -> f w h c').cpu().detach().numpy())
        # _ = utils.create_video(canny_to_save, 4, path="ddxk.mp4", watermark=None)

        f, _, h, w = video.shape
        self.generator.manual_seed(seed)
        latents = torch.randn((1, 4, h//8, w//8), dtype=self.dtype,
                              device=self.device, generator=self.generator)
        latents = latents.repeat(f, 1, 1, 1)
        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                height=h,
                                width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                eta=eta,
                                latents=latents,
                                seed=seed,
                                output_type='numpy',
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                merging_ratio=merging_ratio,
                                )
        return utils.create_video(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_controlnet_depth(self,
                                 video_path,
                                 prompt,
                                 chunk_size=8,
                                 watermark='Picsart AI Research',
                                 merging_ratio=0.0,
                                 num_inference_steps=20,
                                 controlnet_conditioning_scale=1.0,
                                 guidance_scale=9.0,
                                 seed=42,
                                 eta=0.0,
                                 resolution=512,
                                 use_cf_attn=True,
                                 save_path=None):
        print("Module Depth")
        video_path = gradio_utils.edge_path_to_video_path(video_path)
        if self.model_type != ModelType.ControlNetDepth:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth")
            self.set_model(ModelType.ControlNetDepth,
                           model_id="runwayml/stable-diffusion-v1-5", controlnet=controlnet)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.controlnet_attn_proc)
                self.pipe.controlnet.set_attn_processor(
                    processor=self.controlnet_attn_proc)

        added_prompt = 'best quality, extremely detailed'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False)
        control = utils.pre_process_depth(
            video).to(self.device).to(self.dtype)

        # depth_map_to_save = list(rearrange(control, 'f c w h -> f w h c').cpu().detach().numpy())
        # _ = utils.create_video(depth_map_to_save, 4, path="ddxk.mp4", watermark=None)

        f, _, h, w = video.shape
        self.generator.manual_seed(seed)
        latents = torch.randn((1, 4, h//8, w//8), dtype=self.dtype,
                              device=self.device, generator=self.generator)
        latents = latents.repeat(f, 1, 1, 1)
        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                height=h,
                                width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                eta=eta,
                                latents=latents,
                                seed=seed,
                                output_type='numpy',
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                merging_ratio=merging_ratio,
                                )
        return utils.create_video(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_controlnet_pose(self,
                                video_path,
                                prompt,
                                chunk_size=8,
                                watermark='Picsart AI Research',
                                merging_ratio=0.0,
                                num_inference_steps=20,
                                controlnet_conditioning_scale=1.0,
                                guidance_scale=9.0,
                                seed=42,
                                eta=0.0,
                                resolution=512,
                                use_cf_attn=True,
                                save_path=None):
        print("Module Pose")
        video_path = gradio_utils.motion_to_video_path(video_path)
        if self.model_type != ModelType.ControlNetPose:
            controlnet = ControlNetModel.from_pretrained(
                "fusing/stable-diffusion-v1-5-controlnet-openpose")
            self.set_model(ModelType.ControlNetPose,
                           model_id="runwayml/stable-diffusion-v1-5", controlnet=controlnet)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.controlnet_attn_proc)
                self.pipe.controlnet.set_attn_processor(
                    processor=self.controlnet_attn_proc)

        video_path = gradio_utils.motion_to_video_path(
            video_path) if 'Motion' in video_path else video_path

        added_prompt = 'best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic'

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False, output_fps=4)
        control = utils.pre_process_pose(
            video, apply_pose_detect=False).to(self.device).to(self.dtype)
        f, _, h, w = video.shape
        self.generator.manual_seed(seed)
        latents = torch.randn((1, 4, h//8, w//8), dtype=self.dtype,
                              device=self.device, generator=self.generator)
        latents = latents.repeat(f, 1, 1, 1)
        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                height=h,
                                width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                eta=eta,
                                latents=latents,
                                seed=seed,
                                output_type='numpy',
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                merging_ratio=merging_ratio,
                                )
        return utils.create_gif(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_controlnet_canny_db(self,
                                    db_path,
                                    video_path,
                                    prompt,
                                    chunk_size=8,
                                    watermark='Picsart AI Research',
                                    merging_ratio=0.0,
                                    num_inference_steps=20,
                                    controlnet_conditioning_scale=1.0,
                                    guidance_scale=9.0,
                                    seed=42,
                                    eta=0.0,
                                    low_threshold=100,
                                    high_threshold=200,
                                    resolution=512,
                                    use_cf_attn=True,
                                    save_path=None):
        print("Module Canny_DB")
        db_path = gradio_utils.get_model_from_db_selection(db_path)
        video_path = gradio_utils.get_video_from_canny_selection(video_path)
        # Load db and controlnet weights
        if 'db_path' not in self.states or db_path != self.states['db_path']:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny")
            self.set_model(ModelType.ControlNetCannyDB,
                           model_id=db_path, controlnet=controlnet)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            self.states['db_path'] = db_path

        if use_cf_attn:
            self.pipe.unet.set_attn_processor(
                processor=self.controlnet_attn_proc)
            self.pipe.controlnet.set_attn_processor(
                processor=self.controlnet_attn_proc)

        added_prompt = 'best quality, extremely detailed'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False)
        control = utils.pre_process_canny(
            video, low_threshold, high_threshold).to(self.device).to(self.dtype)
        f, _, h, w = video.shape
        self.generator.manual_seed(seed)
        latents = torch.randn((1, 4, h//8, w//8), dtype=self.dtype,
                              device=self.device, generator=self.generator)
        latents = latents.repeat(f, 1, 1, 1)
        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                height=h,
                                width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                eta=eta,
                                latents=latents,
                                seed=seed,
                                output_type='numpy',
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                merging_ratio=merging_ratio,
                                )
        return utils.create_gif(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_pix2pix(self,
                        video,
                        prompt,
                        resolution=512,
                        seed=0,
                        image_guidance_scale=1.0,
                        start_t=0,
                        end_t=-1,
                        out_fps=-1,
                        chunk_size=8,
                        watermark='Picsart AI Research',
                        merging_ratio=0.0,
                        use_cf_attn=True,
                        save_path=None,):
        print("Module Pix2Pix")
        if self.model_type != ModelType.Pix2Pix_Video:
            self.set_model(ModelType.Pix2Pix_Video,
                           model_id="timbrooks/instruct-pix2pix")
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.pix2pix_attn_proc)
        video, fps = utils.prepare_video(
            video, resolution, self.device, self.dtype, True, start_t, end_t, out_fps)
        self.generator.manual_seed(seed)
        result = self.inference(image=video,
                                prompt=prompt,
                                seed=seed,
                                output_type='numpy',
                                num_inference_steps=50,
                                image_guidance_scale=image_guidance_scale,
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                merging_ratio=merging_ratio
                                )
        return utils.create_video(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_text2video(self,
                           prompt,
                           model_name="dreamlike-art/dreamlike-photoreal-2.0",
                           motion_field_strength_x=12,
                           motion_field_strength_y=12,
                           t0=44,
                           t1=47,
                           n_prompt="",
                           chunk_size=8,
                           video_length=8,
                           watermark='Picsart AI Research',
                           merging_ratio=0.0,
                           seed=0,
                           resolution=512,
                           fps=2,
                           use_cf_attn=True,
                           use_motion_field=True,
                           smooth_bg=False,
                           smooth_bg_strength=0.4,
                           path=None):
        print("Module Text2Video")
        if self.model_type != ModelType.Text2Video or model_name != self.model_name:
            print("Model update")
            unet = UNet2DConditionModel.from_pretrained(
                model_name, subfolder="unet")
            self.set_model(ModelType.Text2Video,
                           model_id=model_name, unet=unet)
<<<<<<< HEAD
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
=======
            def check_model_weights(model):
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN found in {name}")
                    if torch.isinf(param).any():
                        print(f"Inf found in {name}")
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            print("check model weights...")
            check_model_weights(self.pipe.unet)
            check_model_weights(self.pipe.text_encoder)
            check_model_weights(self.pipe.vae)
>>>>>>> 0d598340d0636e7a3ff895ad74b8444225852e19
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.text2video_attn_proc)
        self.generator.manual_seed(seed)

        added_prompt = "high quality, HD, 8K, trending on artstation, high focus, dramatic lighting"
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic'

        prompt = prompt.rstrip()
        if len(prompt) > 0 and (prompt[-1] == "," or prompt[-1] == "."):
            prompt = prompt.rstrip()[:-1]
        prompt = prompt.rstrip()
        prompt = prompt + ", "+added_prompt
        if len(n_prompt) > 0:
            negative_prompt = n_prompt
        else:
            negative_prompt = None

        result = self.inference(prompt=prompt,
                                video_length=video_length,
                                height=resolution,
                                width=resolution,
                                num_inference_steps=50,
                                guidance_scale=7.5,
                                guidance_stop_step=1.0,
                                t0=t0,
                                t1=t1,
                                motion_field_strength_x=motion_field_strength_x,
                                motion_field_strength_y=motion_field_strength_y,
                                use_motion_field=use_motion_field,
                                smooth_bg=smooth_bg,
                                smooth_bg_strength=smooth_bg_strength,
                                seed=seed,
                                output_type='numpy',
                                negative_prompt=negative_prompt,
                                merging_ratio=merging_ratio,
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return utils.create_video(result, fps, path=path, watermark=gradio_utils.logo_name_to_path(watermark))
