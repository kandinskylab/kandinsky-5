import torch
import os
from omegaconf.dictconfig import DictConfig
from ..kandinsky.models.vae import build_vae
from ..kandinsky.models.text_embedders import Kandinsky5TextEmbedder
from ..kandinsky.models.dit import get_dit
from ..kandinsky.generation_utils import generate
from ..kandinsky.i2v_pipeline import resize_image
import folder_paths
from comfy.comfy_types import ComfyNodeABC
from comfy.utils import ProgressBar as pbar
from safetensors.torch import load_file
from omegaconf import OmegaConf
from pathlib import Path
from torchvision import transforms


class Kandinsky5LoadTextEmbedders:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen": (os.listdir(folder_paths.get_folder_paths("text_encoders")[0]), {"default": "qwen2_5_vl_7b_instruct"}),
                "clip": (os.listdir(folder_paths.get_folder_paths("text_encoders")[0]), {"default": "clip_text"}),
                "qwen_quantized": ("BOOLEAN", {"default": False})
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_te"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "return clip and qwen text embedders"

    def load_te(self, qwen, clip, qwen_quantized):
        qwen_path = os.path.join(folder_paths.get_folder_paths("text_encoders")[0],qwen)
        clip_path = os.path.join(folder_paths.get_folder_paths("text_encoders")[0],clip)
        conf = {'qwen': {'checkpoint_path': qwen_path, 'max_length': 256},
            'clip': {'checkpoint_path': clip_path, 'max_length': 77},
        }
        return (Kandinsky5TextEmbedder(DictConfig(conf), device='cpu',quantized_qwen=qwen_quantized),)
class Kandinsky5LoadDiT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit": (folder_paths.get_filename_list("diffusion_models"), ),
            }
        }
    RETURN_TYPES = ("MODEL","CONFIG")
    RETURN_NAMES = ("model","conf")
    FUNCTION = "load_dit"
    CATEGORY = "advanced/loaders"

    DESCRIPTION = "return kandy dit"

    def load_dit(self, dit):
        
        dit_path = folder_paths.get_full_path_or_raise("diffusion_models", dit)
        current_file = Path(__file__)
        parent_directory = current_file.parent.parent
        sec = dit.split("_")[-1].split(".")[0]
        conf = OmegaConf.load(os.path.join(parent_directory,f"configs/config_{sec}_sft.yaml"))
        dit = get_dit(conf.model.dit_params)
        state_dict = load_file(dit_path)
        dit.load_state_dict(state_dict)
        return (dit,conf)
class Kandinsky5TextEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True})
            },
            "optional": {
                "extended_text": ("PROMPT",),
            },
        }
    RETURN_TYPES = ("CONDITION", "CONDITION")
    RETURN_NAMES = ("TEXT", "POOLED")
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, model, prompt, extended_text=None):
        text = extended_text if extended_text is not None else prompt
        device='cuda:0'
        model = model.to(device)
        text_embeds = model.embedder([text], type_of_content='video')
        pooled_embed = model.clip_embedder([text])
        model = model.to('cpu')
        return (text_embeds, pooled_embed)

class Kandinsky5LoadVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": (os.listdir(folder_paths.get_folder_paths("vae")[0]), {"default": "hunyuan_vae"}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_vae"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "return vae"

    def load_vae(self, vae):
        vae_path = os.path.join(folder_paths.get_folder_paths("vae")[0],vae)
        vae = build_vae(DictConfig({'checkpoint_path':vae_path, 'name':'hunyuan'}))
        vae = vae.eval()

        return (vae,)
class expand_prompt(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True})
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("PROMPT","STRING")
    RETURN_NAMES = ("exp_prompt","log")
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = ("expanded prompt",)
    FUNCTION = "expand_prompt"

    CATEGORY = "conditioning"
    DESCRIPTION = "extend prompt with."
    def expand_prompt(self, model, prompt, image=None, device='cuda:0'):
        if image is not None:
            print(image.shape)
            to_pil = transforms.ToPILImage()

            # Convert tensor
            pil_image = to_pil(image.squeeze(0).permute(2,0,1))  # Remove batch dimension
            print("i2v expander")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image
                        },
                        {
                            "type": "text",
                            "text": f"""You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
            Here are some example descriptions from the dataset that the model was trained:
            1. "Create a video showing a nighttime urban driving scene from inside a car. The driver is focused on the road ahead, with the city lights visible through the windshield. The GPS device on the dashboard continues to display navigation information. The camera remains steady, capturing the interior of the car and the changing street view outside as the vehicle moves forward. The background shifts slightly to show different parts of the cityscape, including illuminated buildings and street signs."
            2. "Create a video where the character, dressed in historical attire, is seen holding an umbrella with a logo. The character should move closer to the camera while maintaining a steady pace, keeping the umbrella raised. The background remains consistent with a foggy, outdoor setting, but the focus shifts more towards the character as they approach. The lighting should emphasize the details of the costume and the umbrella, enhancing the dramatic effect."
            3. "Darken the scene while keeping the characters and setting unchanged, emphasizing a serious atmosphere."
            IImportantly! These are just examples from a large training dataset of 20 mln videos.
            Rewrite Prompt: "{prompt}" to get high-quality image to video generation from this image. Pay main attention to information about changes of objects.
            Make prompt dynamic. Answer only with expanded prompt..""",
                        },
                    ],
                }
            ]
        else:
            print("t2v expander")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
            Here are some example descriptions from the dataset that the model was trained:
            1. "In a dimly lit room with a cluttered background, papers are pinned to the wall and various objects rest on a desk. Three men stand present: one wearing a red sweater, another in a black sweater, and the third in a gray shirt. The man in the gray shirt speaks and makes hand gestures, while the other two men look forward. The camera remains stationary, focusing on the three men throughout the sequence. A gritty and realistic visual style prevails, marked by a greenish tint that contributes to a moody atmosphere. Low lighting casts shadows, enhancing the tense mood of the scene."
            2. "In an office setting, a man sits at a desk wearing a gray sweater and seated in a black office chair. A wooden cabinet with framed pictures stands beside him, alongside a small plant and a lit desk lamp. Engaged in a conversation, he makes various hand gestures to emphasize his points. His hands move in different positions, indicating different ideas or points. The camera remains stationary, focusing on the man throughout. Warm lighting creates a cozy atmosphere. The man appears to be explaining something. The overall visual style is professional and polished, suitable for a business or educational context."
            3. "A person works on a wooden object resembling a sunburst pattern, holding it in their left hand while using their right hand to insert a thin wire into the gaps between the wooden pieces. The background features a natural outdoor setting with greenery and a tree trunk visible. The camera stays focused on the hands and the wooden object throughout, capturing the detailed process of assembling the wooden structure. The person carefully threads the wire through the gaps, ensuring the wooden pieces are securely fastened together. The scene unfolds with a naturalistic and instructional style, emphasizing the craftsmanship and the methodical steps taken to complete the task."
            IImportantly! These are just examples from a large training dataset of 200 million videos.
            Rewrite Prompt: "{prompt}" to get high-quality video generation. Answer only with expanded prompt.""",
                        },
                    ],
                }
            ]
        model = model.to(device)
        text = model.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = model.embedder.processor(
            text=[text],
            images=None if image is None else [pil_image],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.embedder.model.device)
        generated_ids = model.embedder.model.generate(
            **inputs, max_new_tokens=256
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = model.embedder.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(output_text[0])
        model = model.to('cpu')
        return (output_text[0],str(output_text[0]))
class Kandinsky5Generate(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "config": ("CONFIG", {"tooltip": "Config of model and generation."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "width": ("INT", {"default": 768, "min": 512, "max": 768, "tooltip": "width of video."}),
                "height": ("INT", {"default": 512, "min": 512, "max": 768, "tooltip": "height of video."}),
                "length": ("INT", {"default": 121, "min": 5, "max": 241, "tooltip": "lenght of video."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "scheduler_scale":("FLOAT", {"default": 10.0, "min": 1.0, "max": 25.0, "step":0.1, "round": 0.01, "tooltip": "scheduler scale"}),
                "precision": (["float16", "bfloat16"], {"default": "bfloat16"}),
                "positive_emb": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "positive_clip": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "negative_emb": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative_clip": ("CONDITION", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
            },
            "optional": {
                "image_latent": ("LATENT",),
            },
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, config, steps, width, height, length, cfg, precision, positive_emb, positive_clip, negative_emb, negative_clip, scheduler_scale, image_latent=None):
        bs = 1
        device = 'cuda:0'
        model = model.to(device)
        patch_size = (1, 2, 2)
        autocast_type = torch.bfloat16 if precision=='bfloat16' else torch.float16 
        dim = config.model.dit_params.in_visual_dim
        if image_latent is not None:
            length, height, width = 1 + (length - 1)//4, image_latent.shape[1], image_latent.shape[2]
        else:
            length, height, width = 1 + (length - 1)//4, height // 8, width // 8
        bs_text_embed, text_cu_seqlens, attention_mask = positive_emb
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = negative_emb
        attention_mask  = attention_mask.bool()
        null_attention_mask  = null_attention_mask.bool()
        text_embed = {"text_embeds": bs_text_embed, "pooled_embed": positive_clip }
        null_embed = {"text_embeds": bs_null_text_embed, "pooled_embed": negative_clip }

        visual_rope_pos = [
            torch.arange(length // patch_size[0]),
            torch.arange(height // patch_size[1]),
            torch.arange(width // patch_size[2])
        ]
        text_rope_pos = torch.cat([torch.arange(end) for end in torch.diff(text_cu_seqlens).cpu()])
        null_text_rope_pos = torch.cat([torch.arange(end) for end in torch.diff(null_text_cu_seqlens).cpu()])
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=autocast_type):
                latent_visual = generate(
                    model, device, (bs * length, height, width, dim), steps, 
                    text_embed, null_embed,
                    visual_rope_pos, text_rope_pos, null_text_rope_pos,
                    cfg, scheduler_scale, 
                    image_latent, 
                    config,
                    attention_mask=attention_mask,
                    null_attention_mask=null_attention_mask
                )
                if image_latent is not None:
                    image_latent = image_latent.to(device=latent_visual.device, dtype=latent_visual.dtype)
                    latent_visual[:1] = image_latent
        model = model.to('cpu')
        return (latent_visual,)

class Kandinsky5VAEDecode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "vae."}),
                "latent": ("LATENT", {"tooltip": "latent."}),}
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)
    FUNCTION = "decode"
    CATEGORY = "latent"
    DESCRIPTION = "Decodes latent images back into pixel space images."

    def decode(self, model, latent):
        device = 'cuda:0'
        model = model.to(device)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                bs = 1
                images = latent.reshape(bs, -1, latent.shape[-3], latent.shape[-2], latent.shape[-1])# bs, t, h, w, c
                # shape for decode: bs, c, t, h, w
                images = (images / 0.476986).permute(0, 4, 1, 2, 3)
                images = model.decode(images).sample
                if not isinstance(images, torch.Tensor):
                    images = images.sample
                images = ((images.clamp(-1., 1.) + 1.) * 0.5)#.to(torch.uint8)
        images = images[0].float().permute(1, 2, 3, 0)
        model = model.to('cpu')
        return (images,)

class Kandinsky5VAEImageEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "vae."}),
                "image": ("IMAGE", {"tooltip": "image."}),}
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The encoded latent.",)
    FUNCTION = "encode"
    CATEGORY = "latent"
    DESCRIPTION = "Encodes image to latent."

    def encode(self, model, image):
        device = 'cuda:0'
        model = model.to(device)
        image, k = resize_image(image.permute(0,3,1,2), max_area=512*768)
        image = image * 2.0 - 1.

        with torch.no_grad():
            image = image.to(device=device, dtype=torch.float16).unsqueeze(0).transpose(1,2)
            print(image.shape)
            lat_image = model.encode(image, opt_tiling=False).latent_dist.sample().squeeze(0).permute(1, 2, 3, 0)
            lat_image = lat_image * 0.476986

        model = model.to('cpu')
        return (lat_image,)

NODE_CLASS_MAPPINGS = {
    "Kandinsky5LoadTextEmbedders": Kandinsky5LoadTextEmbedders,
    "Kandinsky5TextEncode": Kandinsky5TextEncode,
    "Kandinsky5Generate": Kandinsky5Generate,
    "Kandinsky5LoadVAE": Kandinsky5LoadVAE,
    "Kandinsky5VAEDecode": Kandinsky5VAEDecode,
    "Kandinsky5VAEImageEncode":Kandinsky5VAEImageEncode,
    "Kandinsky5LoadDiT": Kandinsky5LoadDiT,
    "expand_prompt": expand_prompt
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Kandinsky5LoadTextEmbedders": "Kandinsky5LoadTextEmbedders",
    "Kandinsky5TextEncode": "Kandinsky5TextEncode",
    "Kandinsky5Generate": "Kandinsky5Generate",
    "Kandinsky5LoadVAE": "Kandinsky5LoadVAE",
    "Kandinsky5VAEDecode": "Kandinsky5VAEDecode",
    "Kandinsky5VAEImageEncode": "Kandinsky5VAEImageEncode",
    "Kandinsky5LoadDiT": "Kandinsky5LoadDiT",
    "expand_prompt": "expand_prompt"

}
