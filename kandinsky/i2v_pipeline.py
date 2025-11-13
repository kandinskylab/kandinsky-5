from math import floor, sqrt
from typing import Union

import transformers
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
from PIL import Image

from .generation_utils import generate_sample_i2v

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True

MAX_AREA = 768*512

def resize_image(image, max_area, alignment=16):
    """
    Resize image to fit within max_area while maintaining aspect ratio.

    Args:
        image: Input tensor image
        max_area: Maximum allowed area (height * width)
        alignment: Pixel alignment requirement (default 16, use 128 for NABLA attention)
    """
    h, w = image.shape[2:]
    area = h * w
    k = sqrt(max_area / area) / alignment
    new_h = int(floor(h * k) * alignment)
    new_w = int(floor(w * k) * alignment)
    return F.resize(image, (new_h, new_w)), k


def get_first_frame_from_image(image, vae, device, alignment=16):
    """
    Load and encode an image to latent space.

    Args:
        image: Path to image or PIL Image
        vae: VAE model
        device: Device to use
        alignment: Pixel alignment for resizing (use 128 for NABLA attention)
    """
    if isinstance(image, str):
        pil_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError(f"unknown image type: {type(image)}")

    image = F.pil_to_tensor(pil_image).unsqueeze(0)
    image, k = resize_image(image, max_area=MAX_AREA, alignment=alignment)
    image = image / 127.5 - 1.

    with torch.no_grad():
        image = image.to(device=device, dtype=torch.float16).transpose(0, 1).unsqueeze(0)
        lat_image = vae.encode(image, opt_tiling=False).latent_dist.sample().squeeze(0).permute(1, 2, 3, 0)
        lat_image = lat_image * vae.config.scaling_factor

    return pil_image, lat_image, k


class Kandinsky5I2VPipeline:
    def __init__(
        self,
        device_map: Union[
            str, torch.device, dict
        ],  # {"dit": cuda:0, "vae": cuda:1, "text_embedder": cuda:1 }
        dit,
        text_embedder,
        vae,
        local_dit_rank: int = 0,
        world_size: int = 1,
        conf = None,
        offload: bool = False,
    ):
        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae

        self.device_map = device_map
        self.local_dit_rank = local_dit_rank
        self.world_size = world_size
        self.conf = conf
        self.num_steps = conf.model.num_steps
        self.guidance_weight = conf.model.guidance_weight

        self.offload = offload


    def __call__(
        self,
        text: str,
        image: Union[str, Image.Image],
        time_length: int = 5,  # time in seconds 0 if you want generate image
        seed: int = None,
        num_steps: int = None,
        guidance_weight: float = None,
        scheduler_scale: float = 10.0,
        negative_caption: str = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
    ):
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight
        # SEED
        if seed is None:
            if self.local_dit_rank == 0:
                seed = torch.randint(2**32 - 1, (1,)).to(self.local_dit_rank)
            else:
                seed = torch.empty((1,), dtype=torch.int64).to(self.local_dit_rank)

            if self.world_size > 1:
                torch.distributed.broadcast(seed, 0)

            seed = seed.item()

        # PREPARATION
        num_frames = 1 if time_length == 0 else time_length * 24 // 4 + 1

        # For NABLA attention with fractal flattening, need 128-pixel alignment
        # For other attention types, 16-pixel alignment is sufficient
        alignment = 128 if getattr(self.conf.model.attention, 'type', 'flash') == 'nabla' else 16

        if self.offload:
            self.vae = self.vae.to(self.device_map["vae"], non_blocking=True)
        image, image_lat, k = get_first_frame_from_image(image, self.vae, self.device_map["vae"], alignment=alignment)
        if self.offload:
            self.vae = self.vae.to("cpu", non_blocking=True)

        caption = text
        if expand_prompts:
            transformers.set_seed(seed)
            if self.local_dit_rank == 0:
                if self.offload:
                    self.text_embedder = self.text_embedder.to(self.device_map["text_embedder"])
                caption = self.text_embedder.embedder.expand_text_prompt(caption, image, device=self.device_map["text_embedder"])
            if self.world_size > 1:
                caption = [caption]
                torch.distributed.broadcast_object_list(caption, 0)
                caption = caption[0]

        height, width = image_lat.shape[1:3]
        shape = (1, num_frames, height, width, 16)

        # GENERATION
        images = generate_sample_i2v(
            shape,
            caption,
            self.dit,
            self.vae,
            self.conf,
            text_embedder=self.text_embedder,
            images=image_lat,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            seed=seed,
            device=self.device_map["dit"],
            vae_device=self.device_map["vae"],
            progress=progress,
            offload=self.offload
        )
        torch.cuda.empty_cache()

        if k > 16:
            h, w = images.shape[-2:]
            images = F.resize(images[0], (int(h / k / 16), int(w / k / 16)))

        # RESULTS
        if self.local_dit_rank == 0:
            if time_length == 0:
                return_images = []
                for image in images.squeeze(2).cpu():
                    return_images.append(ToPILImage()(image))
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(return_images):
                        for path, image in zip(save_path, return_images):
                            image.save(path)
                return return_images
            else:
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(images):
                        for path, video in zip(save_path, images):
                            torchvision.io.write_video(
                                path,
                                video.float().permute(1, 2, 3, 0).cpu().numpy(),
                                fps=24,
                                options={"crf": "5"},
                            )
                return images
