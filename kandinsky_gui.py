"""
Kandinsky 5 Gradio GUI
A simple interface for generating videos with Kandinsky models
"""

import os
import time
import warnings
import logging
import threading
import gradio as gr
import torch
from pathlib import Path
from datetime import datetime

from kandinsky import get_T2V_pipeline, get_I2V_pipeline, get_I2V_pipeline_with_block_swap

# Global state
current_pipeline = None
current_config = None
stop_event = threading.Event()

# Disable warnings
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

# Available configurations
CONFIGS = {
    "5s SFT (Recommended)": "./configs/config_5s_sft.yaml",
    "5s Pretrain": "./configs/config_5s_pretrain.yaml",
    "5s Distilled": "./configs/config_5s_distil.yaml",
    "5s No CFG": "./configs/config_5s_nocfg.yaml",
    "5s I2V": "./configs/config_5s_i2v.yaml",
    "5s I2V Pro 20B": "./configs/config_5s_i2v_pro_20b.yaml",
    "10s SFT": "./configs/config_10s_sft.yaml",
    "10s Pretrain": "./configs/config_10s_pretrain.yaml",
    "10s Distilled": "./configs/config_10s_distil.yaml",
    "10s No CFG": "./configs/config_10s_nocfg.yaml",
}

# Output directory
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_pipeline(config_name, enable_offload, quantized_qwen, enable_block_swap, blocks_in_memory, attention_engine, progress=gr.Progress()):
    """Load the pipeline with specified configuration"""
    global current_pipeline, current_config

    progress(0, desc="Loading model...")

    config_path = CONFIGS[config_name]

    # Check if we need to reload
    pipeline_key = (config_path, enable_offload, quantized_qwen, enable_block_swap, blocks_in_memory, attention_engine)

    if current_pipeline is not None and current_config == pipeline_key:
        return "Model already loaded!"

    # Clear existing pipeline
    current_pipeline = None
    torch.cuda.empty_cache()

    device_map = {
        "dit": torch.device('cuda:0'),
        "vae": torch.device('cuda:0'),
        "text_embedder": torch.device('cuda:0')
    }

    try:
        is_i2v = "i2v" in config_path

        if is_i2v and enable_block_swap:
            progress(0.3, desc="Loading I2V pipeline with block swapping...")
            current_pipeline = get_I2V_pipeline_with_block_swap(
                device_map=device_map,
                conf_path=config_path,
                offload=enable_offload,
                quantized_qwen=quantized_qwen,
                attention_engine=attention_engine,
                blocks_in_memory=blocks_in_memory,
                enable_block_swap=True,
            )
        elif is_i2v:
            progress(0.3, desc="Loading I2V pipeline...")
            current_pipeline = get_I2V_pipeline(
                device_map=device_map,
                conf_path=config_path,
                offload=enable_offload,
                quantized_qwen=quantized_qwen,
                attention_engine=attention_engine,
            )
        else:
            progress(0.3, desc="Loading T2V pipeline...")
            current_pipeline = get_T2V_pipeline(
                device_map=device_map,
                conf_path=config_path,
                offload=enable_offload,
                quantized_qwen=quantized_qwen,
                attention_engine=attention_engine,
            )

        current_config = pipeline_key
        progress(1.0, desc="Model loaded successfully!")
        return f"✓ Model loaded: {config_name}"

    except Exception as e:
        return f"❌ Error loading model: {str(e)}"


def generate_t2v(prompt, negative_prompt, width, height, duration, num_steps,
                 guidance_weight, scheduler_scale, seed, expand_prompt,
                 config_name, enable_offload, quantized_qwen, enable_block_swap,
                 blocks_in_memory, attention_engine, progress=gr.Progress()):
    """Generate text-to-video"""
    global current_pipeline, stop_event

    stop_event.clear()

    # Load pipeline if needed
    load_status = load_pipeline(config_name, enable_offload, quantized_qwen,
                                enable_block_swap, blocks_in_memory, attention_engine, progress)

    if "Error" in load_status:
        return None, load_status

    if "i2v" in CONFIGS[config_name]:
        return None, "❌ Please select a T2V model (not I2V) for text-to-video generation"

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"t2v_{timestamp}.mp4")

    try:
        progress(0, desc="Generating video...")
        start_time = time.perf_counter()

        result = current_pipeline(
            prompt,
            time_length=duration,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance_weight=guidance_weight if guidance_weight > 0 else None,
            scheduler_scale=scheduler_scale,
            expand_prompts=expand_prompt,
            save_path=output_path,
            seed=seed
        )

        elapsed = time.perf_counter() - start_time

        if stop_event.is_set():
            return None, "⏹ Generation stopped by user"

        progress(1.0, desc="Video generated!")

        status = f"✓ Video generated in {elapsed:.1f}s\nSaved to: {output_path}"
        return output_path, status

    except Exception as e:
        return None, f"❌ Error during generation: {str(e)}"


def generate_i2v(prompt, negative_prompt, image, duration, num_steps,
                guidance_weight, scheduler_scale, seed, expand_prompt,
                config_name, enable_offload, quantized_qwen, enable_block_swap,
                blocks_in_memory, attention_engine, progress=gr.Progress()):
    """Generate image-to-video"""
    global current_pipeline, stop_event

    stop_event.clear()

    if image is None:
        return None, "❌ Please provide an input image"

    # Load pipeline if needed
    load_status = load_pipeline(config_name, enable_offload, quantized_qwen,
                                enable_block_swap, blocks_in_memory, attention_engine, progress)

    if "Error" in load_status:
        return None, load_status

    if "i2v" not in CONFIGS[config_name]:
        return None, "❌ Please select an I2V model for image-to-video generation"

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"i2v_{timestamp}.mp4")

    try:
        progress(0, desc="Generating video...")
        start_time = time.perf_counter()

        result = current_pipeline(
            prompt,
            image=image,
            time_length=duration,
            num_steps=num_steps,
            guidance_weight=guidance_weight if guidance_weight > 0 else None,
            scheduler_scale=scheduler_scale,
            expand_prompts=expand_prompt,
            save_path=output_path,
            seed=seed
        )

        elapsed = time.perf_counter() - start_time

        if stop_event.is_set():
            return None, "⏹ Generation stopped by user"

        progress(1.0, desc="Video generated!")

        status = f"✓ Video generated in {elapsed:.1f}s\nSaved to: {output_path}"
        return output_path, status

    except Exception as e:
        return None, f"❌ Error during generation: {str(e)}"


def stop_generation():
    """Stop the current generation"""
    global stop_event
    stop_event.set()
    return "⏹ Stopping generation..."


def random_seed():
    """Generate a random seed"""
    import random
    return random.randint(0, 2**32 - 1)


# Custom CSS styling (inspired by H1111)
custom_css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
}

.generate-btn {
    background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

.generate-btn:hover {
    background: linear-gradient(90deg, #45a049 0%, #3d8b40 100%) !important;
}

.stop-btn {
    background: linear-gradient(90deg, #f44336 0%, #da190b 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

.load-btn {
    background: linear-gradient(90deg, #2196F3 0%, #0b7dda 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

.tabs {
    margin-top: 10px;
}

.tab-nav button {
    font-size: 16px !important;
    font-weight: 600 !important;
}

#output-video {
    min-height: 400px;
}

.status-box {
    font-family: monospace;
    padding: 10px;
    border-radius: 5px;
    background-color: #f5f5f5;
}
"""


# Build the UI
with gr.Blocks(css=custom_css, title="Kandinsky 5 GUI") as demo:
    gr.Markdown("# 🎨 Kandinsky 5 Video Generation")
    gr.Markdown("Generate high-quality videos from text prompts or images using Kandinsky models")

    with gr.Row():
        with gr.Column(scale=1):
            # Model configuration section
            gr.Markdown("### 🔧 Model Configuration")

            config_dropdown = gr.Dropdown(
                choices=list(CONFIGS.keys()),
                value="5s SFT (Recommended)",
                label="Model",
                info="Select the model configuration to use"
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                enable_offload = gr.Checkbox(
                    label="Enable Model Offloading",
                    value=False,
                    info="Offload models to CPU when not in use (saves VRAM)"
                )

                quantized_qwen = gr.Checkbox(
                    label="Use Quantized Qwen (4-bit)",
                    value=True,
                    info="Use 4-bit quantized text embedder (saves ~12GB VRAM)"
                )

                enable_block_swap = gr.Checkbox(
                    label="Enable Block Swapping",
                    value=False,
                    info="For 20B model: swap transformer blocks to fit in limited VRAM"
                )

                blocks_in_memory = gr.Slider(
                    minimum=2,
                    maximum=12,
                    value=6,
                    step=1,
                    label="Blocks in Memory",
                    info="Number of transformer blocks to keep in GPU memory (only for block swapping)"
                )

                attention_engine = gr.Dropdown(
                    choices=["auto", "flash_attention_2", "flash_attention_3", "sdpa", "sage"],
                    value="auto",
                    label="Attention Engine",
                    info="Attention mechanism to use"
                )

            load_btn = gr.Button("🔄 Load/Reload Model", elem_classes="load-btn", size="sm")
            load_status = gr.Textbox(label="Load Status", lines=2, interactive=False)

    with gr.Tabs(elem_classes="tabs"):
        # T2V Tab
        with gr.Tab("📝 Text-to-Video (T2V)"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2v_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the video you want to generate...",
                        value="A dragon soaring through the sky at sunset",
                        lines=3
                    )

                    t2v_negative = gr.Textbox(
                        label="Negative Prompt",
                        value="Static, 2D cartoon, worst quality, low quality, ugly, deformed",
                        lines=2
                    )

                    with gr.Row():
                        t2v_width = gr.Dropdown(
                            choices=[512, 768],
                            value=768,
                            label="Width"
                        )
                        t2v_height = gr.Dropdown(
                            choices=[512, 768],
                            value=512,
                            label="Height"
                        )

                    with gr.Row():
                        t2v_duration = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Duration (seconds)"
                        )
                        t2v_steps = gr.Slider(
                            minimum=20,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Steps"
                        )

                    with gr.Row():
                        t2v_guidance = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=7.5,
                            step=0.1,
                            label="Guidance Weight"
                        )
                        t2v_scheduler = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.1,
                            label="Scheduler Scale"
                        )

                    with gr.Row():
                        t2v_seed = gr.Number(
                            label="Seed",
                            value=1137,
                            precision=0
                        )
                        t2v_random_seed = gr.Button("🎲", size="sm")
                        t2v_expand = gr.Checkbox(
                            label="Expand Prompt",
                            value=True
                        )

                    with gr.Row():
                        t2v_generate_btn = gr.Button("🎬 Generate Video", elem_classes="generate-btn", size="lg")
                        t2v_stop_btn = gr.Button("⏹ Stop", elem_classes="stop-btn", size="lg")

                with gr.Column(scale=1):
                    t2v_output = gr.Video(label="Generated Video", elem_id="output-video")
                    t2v_status = gr.Textbox(label="Status", lines=3, interactive=False, elem_classes="status-box")

        # I2V Tab
        with gr.Tab("🖼️ Image-to-Video (I2V)"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2v_image = gr.Image(
                        label="Input Image",
                        type="filepath",
                        sources=["upload", "clipboard"]
                    )

                    i2v_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the motion/animation...",
                        value="The dragon breathes fire",
                        lines=3
                    )

                    i2v_negative = gr.Textbox(
                        label="Negative Prompt",
                        value="Static, 2D cartoon, worst quality, low quality, ugly, deformed",
                        lines=2
                    )

                    with gr.Row():
                        i2v_duration = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Duration (seconds)"
                        )
                        i2v_steps = gr.Slider(
                            minimum=20,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Steps"
                        )

                    with gr.Row():
                        i2v_guidance = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=7.5,
                            step=0.1,
                            label="Guidance Weight"
                        )
                        i2v_scheduler = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.1,
                            label="Scheduler Scale"
                        )

                    with gr.Row():
                        i2v_seed = gr.Number(
                            label="Seed",
                            value=1137,
                            precision=0
                        )
                        i2v_random_seed = gr.Button("🎲", size="sm")
                        i2v_expand = gr.Checkbox(
                            label="Expand Prompt",
                            value=True
                        )

                    with gr.Row():
                        i2v_generate_btn = gr.Button("🎬 Generate Video", elem_classes="generate-btn", size="lg")
                        i2v_stop_btn = gr.Button("⏹ Stop", elem_classes="stop-btn", size="lg")

                with gr.Column(scale=1):
                    i2v_output = gr.Video(label="Generated Video", elem_id="output-video")
                    i2v_status = gr.Textbox(label="Status", lines=3, interactive=False, elem_classes="status-box")

    # Footer
    gr.Markdown("---")
    gr.Markdown("💡 **Tips:** Use quantized Qwen to save VRAM. Enable block swapping for the 20B model on GPUs with limited memory.")

    # Event handlers
    load_btn.click(
        fn=load_pipeline,
        inputs=[config_dropdown, enable_offload, quantized_qwen, enable_block_swap,
                blocks_in_memory, attention_engine],
        outputs=[load_status]
    )

    # T2V handlers
    t2v_random_seed.click(fn=random_seed, outputs=[t2v_seed])
    t2v_generate_btn.click(
        fn=generate_t2v,
        inputs=[t2v_prompt, t2v_negative, t2v_width, t2v_height, t2v_duration,
                t2v_steps, t2v_guidance, t2v_scheduler, t2v_seed, t2v_expand,
                config_dropdown, enable_offload, quantized_qwen, enable_block_swap,
                blocks_in_memory, attention_engine],
        outputs=[t2v_output, t2v_status]
    )
    t2v_stop_btn.click(fn=stop_generation, outputs=[t2v_status])

    # I2V handlers
    i2v_random_seed.click(fn=random_seed, outputs=[i2v_seed])
    i2v_generate_btn.click(
        fn=generate_i2v,
        inputs=[i2v_prompt, i2v_negative, i2v_image, i2v_duration,
                i2v_steps, i2v_guidance, i2v_scheduler, i2v_seed, i2v_expand,
                config_dropdown, enable_offload, quantized_qwen, enable_block_swap,
                blocks_in_memory, attention_engine],
        outputs=[i2v_output, i2v_status]
    )
    i2v_stop_btn.click(fn=stop_generation, outputs=[i2v_status])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
