import os
from typing import Optional, Union

import numpy as np
import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from .models.dit import get_dit
from .models.dit_block_swap import get_dit_with_block_swap
from .models.text_embedders import get_text_embedder
from .models.vae import build_vae
from .models.parallelize import parallelize_dit
from .i2v_pipeline import Kandinsky5I2VPipeline
from .t2v_pipeline import Kandinsky5T2VPipeline
from .magcache_utils import set_magcache_params

from PIL import Image
from safetensors.torch import load_file

torch._dynamo.config.suppress_errors = True


def get_T2V_pipeline(
    device_map: Union[str, torch.device, dict],
    resolution: int = 512,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    offload: bool = False,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = True,
    attention_engine: str = "auto",
) -> Kandinsky5T2VPipeline:
    assert resolution in [512]

    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    try:
        local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(
            os.environ["WORLD_SIZE"]
        )
    except:
        local_rank, world_size = 0, 1

    assert not (world_size > 1 and offload), "Offloading available only with not parallel inference"

    if world_size > 1:
        device_mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("tensor_parallel",)
        )
        device_map["dit"] = torch.device(f"cuda:{local_rank}")
        device_map["vae"] = torch.device(f"cuda:{local_rank}")
        device_map["text_embedder"] = torch.device(f"cuda:{local_rank}")

    os.makedirs(cache_dir, exist_ok=True)

    if dit_path is None and conf_path is None:
        dit_path = snapshot_download(
            repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s",
            allow_patterns="model/*",
            local_dir=cache_dir,
        )
        dit_path = os.path.join(cache_dir, "model/kandinsky5lite_t2v_sft_5s.safetensors")

    if vae_path is None and conf_path is None:
        vae_path = snapshot_download(
            repo_id="hunyuanvideo-community/HunyuanVideo",
            allow_patterns="vae/*",
            local_dir=cache_dir,
        )
        vae_path = os.path.join(cache_dir, "vae/")

    if text_encoder_path is None and conf_path is None:
        text_encoder_path = snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=os.path.join(cache_dir, "text_encoder/"),
        )
        text_encoder_path = os.path.join(cache_dir, "text_encoder/")

    if text_encoder2_path is None and conf_path is None:
        text_encoder2_path = snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=os.path.join(cache_dir, "text_encoder2/"),
        )
        text_encoder2_path = os.path.join(cache_dir, "text_encoder2/")

    if conf_path is None:
        conf = get_default_conf(
            dit_path, vae_path, text_encoder_path, text_encoder2_path
        )
    else:
        conf = OmegaConf.load(conf_path)
    conf.model.dit_params.attention_engine = attention_engine

    conf.model.text_embedder.qwen.mode = "t2v"
    text_embedder = get_text_embedder(conf.model.text_embedder, device=device_map["text_embedder"],
                                      quantized_qwen=quantized_qwen, text_token_padding=text_token_padding)
    if not offload: 
        text_embedder = text_embedder.to( device=device_map["text_embedder"]) 
    
    vae = build_vae(conf.model.vae)
    vae = vae.eval()
    if not offload:
        vae = vae.to(device=device_map["vae"]) 

    dit = get_dit(conf.model.dit_params)

    if magcache:
        mag_ratios = conf.magcache.mag_ratios
        num_steps = conf.model.num_steps
        no_cfg = False
        if conf.model.guidance_weight == 1.0:
            no_cfg = True
        set_magcache_params(dit, mag_ratios, num_steps, no_cfg)

    state_dict = load_file(conf.model.checkpoint_path)
    dit.load_state_dict(state_dict, assign=True)

    if not offload:
        dit = dit.to(device_map["dit"])

    if world_size > 1:
        dit = parallelize_dit(dit, device_mesh["tensor_parallel"])

    return Kandinsky5T2VPipeline(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        resolution=resolution,
        local_dit_rank=local_rank,
        world_size=world_size,
        conf=conf,
        offload=offload,
    )

def get_I2V_pipeline(
    device_map: Union[str, torch.device, dict],
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    offload: bool = False,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = True,
    attention_engine: str = "auto",
) -> Kandinsky5T2VPipeline:
    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    try:
        local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(
            os.environ["WORLD_SIZE"]
        )
    except:
        local_rank, world_size = 0, 1

    assert not (world_size > 1 and offload), "Offloading available only with not parallel inference"

    if world_size > 1:
        device_mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("tensor_parallel",)
        )
        device_map["dit"] = torch.device(f"cuda:{local_rank}")
        device_map["vae"] = torch.device(f"cuda:{local_rank}")
        device_map["text_embedder"] = torch.device(f"cuda:{local_rank}")

    os.makedirs(cache_dir, exist_ok=True)

    if dit_path is None and conf_path is None:
        dit_path = snapshot_download(
            repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s",
            allow_patterns="model/*",
            local_dir=cache_dir,
        )
        dit_path = os.path.join(cache_dir, "model/kandinsky5lite_i2v_sft_5s.safetensors")

    if vae_path is None and conf_path is None:
        vae_path = snapshot_download(
            repo_id="hunyuanvideo-community/HunyuanVideo",
            allow_patterns="vae/*",
            local_dir=cache_dir,
        )
        vae_path = os.path.join(cache_dir, "vae/")

    if text_encoder_path is None and conf_path is None:
        text_encoder_path = snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=os.path.join(cache_dir, "text_encoder/"),
        )
        text_encoder_path = os.path.join(cache_dir, "text_encoder/")

    if text_encoder2_path is None and conf_path is None:
        text_encoder2_path = snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=os.path.join(cache_dir, "text_encoder2/"),
        )
        text_encoder2_path = os.path.join(cache_dir, "text_encoder2/")

    if conf_path is None:
        conf = get_default_conf(
            dit_path, vae_path, text_encoder_path, text_encoder2_path
        )
    else:
        conf = OmegaConf.load(conf_path)
    conf.model.dit_params.attention_engine = attention_engine

    conf.model.text_embedder.qwen.mode = "i2v"
    text_embedder = get_text_embedder(conf.model.text_embedder, device=device_map["text_embedder"],
                                      quantized_qwen=quantized_qwen, text_token_padding=text_token_padding)
    if not offload: 
        text_embedder = text_embedder.to( device=device_map["text_embedder"]) 
    
    vae = build_vae(conf.model.vae)
    vae = vae.eval()
    if not offload:
        vae = vae.to(device=device_map["vae"]) 

    dit = get_dit(conf.model.dit_params)

    if magcache:
        mag_ratios = conf.magcache.mag_ratios
        num_steps = conf.model.num_steps
        no_cfg = False
        if conf.model.guidance_weight == 1.0:
            no_cfg = True
        set_magcache_params(dit, mag_ratios, num_steps, no_cfg)

    state_dict = load_file(conf.model.checkpoint_path)
    dit.load_state_dict(state_dict, assign=True)

    if not offload:
        dit = dit.to(device_map["dit"])

    if world_size > 1:
        dit = parallelize_dit(dit, device_mesh["tensor_parallel"])

    return Kandinsky5I2VPipeline(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        local_dit_rank=local_rank,
        world_size=world_size,
        conf=conf,
        offload=offload,
    )

def get_default_conf(
    dit_path,
    vae_path,
    text_encoder_path,
    text_encoder2_path,
) -> DictConfig:
    dit_params = {
        "in_visual_dim": 16,
        "out_visual_dim": 16,
        "time_dim": 512,
        "patch_size": [1, 2, 2],
        "model_dim": 1792,
        "ff_dim": 7168,
        "num_text_blocks": 2,
        "num_visual_blocks": 32,
        "axes_dims": [16, 24, 24],
        "visual_cond": True,
        "in_text_dim": 3584,
        "in_text_dim2": 768,
    }

    attention = {
        "type": "flash",
        "causal": False,
        "local": False,
        "glob": False,
        "window": 3,
    }

    vae = {
        "checkpoint_path": vae_path,
        "name": "hunyuan",
    }

    text_embedder = {
        "qwen": {
            "emb_size": 3584,
            "checkpoint_path": text_encoder_path,
            "max_length": 256,
        },
        "clip": {
            "checkpoint_path": text_encoder2_path,
            "emb_size": 768,
            "max_length": 77,
        },
    }

    conf = {
        "model": {
            "checkpoint_path": dit_path,
            "vae": vae,
            "text_embedder": text_embedder,
            "dit_params": dit_params,
            "attention": attention,
            "num_steps": 50,
            "guidance_weight": 5.0,
        },
        "metrics": {"scale_factor": (1, 2, 2)},
        "resolution": 512,
    }

    return DictConfig(conf)


def get_I2V_pipeline_with_block_swap(
    device_map: Union[str, torch.device, dict],
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    offload: bool = False,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = True,
    attention_engine: str = "auto",
    blocks_in_memory: int = 4,
    enable_block_swap: bool = True,
) -> Kandinsky5I2VPipeline:
    """
    Get I2V pipeline with block swapping support for large models (e.g., 20B).

    Args:
        device_map: Device mapping for components
        cache_dir: Directory to cache downloaded weights
        dit_path: Path to DiT checkpoint
        text_encoder_path: Path to text encoder
        text_encoder2_path: Path to secondary text encoder
        vae_path: Path to VAE
        conf_path: Path to config YAML file
        offload: Enable full model offloading
        magcache: Enable MagCache
        quantized_qwen: Use quantized Qwen encoder
        text_token_padding: Enable text token padding
        attention_engine: Attention implementation to use
        blocks_in_memory: Number of transformer blocks to keep in GPU memory
        enable_block_swap: Enable block swapping (set False to disable for debugging)

    Returns:
        Kandinsky5I2VPipeline with block-swapping enabled DiT
    """
    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    try:
        local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(
            os.environ["WORLD_SIZE"]
        )
    except:
        local_rank, world_size = 0, 1

    if enable_block_swap and world_size > 1:
        print("Warning: Block swapping with multi-GPU not fully tested. Disabling block swap.")
        enable_block_swap = False

    if world_size > 1:
        device_mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("tensor_parallel",)
        )
        device_map["dit"] = torch.device(f"cuda:{local_rank}")
        device_map["vae"] = torch.device(f"cuda:{local_rank}")
        device_map["text_embedder"] = torch.device(f"cuda:{local_rank}")

    os.makedirs(cache_dir, exist_ok=True)

    # Load config
    if conf_path is None:
        raise ValueError("For block swap pipeline, conf_path must be specified")

    conf = OmegaConf.load(conf_path)
    conf.model.dit_params.attention_engine = attention_engine

    # CLI parameters take priority over config file
    # Only use config values if CLI parameters are at default values
    # if hasattr(conf, 'block_swap'):
    #     enable_block_swap = conf.block_swap.get('enabled', enable_block_swap)
    #     blocks_in_memory = conf.block_swap.get('blocks_in_memory', blocks_in_memory)

    # Build text embedder
    conf.model.text_embedder.qwen.mode = "i2v"
    text_embedder = get_text_embedder(
        conf.model.text_embedder,
        device=device_map["text_embedder"],
        quantized_qwen=quantized_qwen,
        text_token_padding=text_token_padding
    )
    if not offload:
        text_embedder = text_embedder.to(device=device_map["text_embedder"])

    # Build VAE
    vae = build_vae(conf.model.vae)
    vae = vae.eval()
    if not offload:
        vae = vae.to(device=device_map["vae"])

    # Build DiT with block swapping
    print(f"Building DiT with block swapping: enabled={enable_block_swap}, blocks_in_memory={blocks_in_memory}")
    dit = get_dit_with_block_swap(
        conf.model.dit_params,
        blocks_in_memory=blocks_in_memory,
        enable_block_swap=enable_block_swap
    )

    if magcache:
        if enable_block_swap:
            print("Warning: MagCache with block swapping not tested. Proceed with caution.")
        mag_ratios = conf.magcache.mag_ratios
        num_steps = conf.model.num_steps
        no_cfg = False
        if conf.model.guidance_weight == 1.0:
            no_cfg = True
        set_magcache_params(dit, mag_ratios, num_steps, no_cfg)

    print(f"Loading DiT weights from {conf.model.checkpoint_path}")
    state_dict = load_file(conf.model.checkpoint_path)
    dit.load_state_dict(state_dict, assign=True)

    if not offload:
        dit = dit.to(device_map["dit"])

    if world_size > 1:
        if enable_block_swap:
            print("Warning: Parallelization with block swapping not supported. Skipping parallelization.")
        else:
            dit = parallelize_dit(dit, device_mesh["tensor_parallel"])

    return Kandinsky5I2VPipeline(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        local_dit_rank=local_rank,
        world_size=world_size,
        conf=conf,
        offload=offload,
    )
