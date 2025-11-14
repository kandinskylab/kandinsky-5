"""
Block-swapping enabled DiT for running 20B model on 48GB VRAM
Keeps only a subset of transformer blocks in GPU memory at a time
"""
import torch
from torch import nn
from .dit import DiffusionTransformer3D, TransformerEncoderBlock, TransformerDecoderBlock


class DiffusionTransformer3DBlockSwap(DiffusionTransformer3D):
    """
    Extended DiT with block swapping support for visual transformer blocks.

    For the 20B model with 60 visual blocks, we keep only `blocks_in_memory` blocks
    in GPU memory at once, swapping them in/out from CPU as needed.
    """
    def __init__(
        self,
        in_visual_dim=4,
        in_text_dim=3584,
        in_text_dim2=768,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_text_blocks=2,
        num_visual_blocks=32,
        axes_dims=(16, 24, 24),
        visual_cond=False,
        attention_engine="auto",
        blocks_in_memory=4,  # Number of visual blocks to keep in GPU memory
        enable_block_swap=False,  # Enable block swapping
    ):
        # Initialize parent class
        super().__init__(
            in_visual_dim=in_visual_dim,
            in_text_dim=in_text_dim,
            in_text_dim2=in_text_dim2,
            time_dim=time_dim,
            out_visual_dim=out_visual_dim,
            patch_size=patch_size,
            model_dim=model_dim,
            ff_dim=ff_dim,
            num_text_blocks=num_text_blocks,
            num_visual_blocks=num_visual_blocks,
            axes_dims=axes_dims,
            visual_cond=visual_cond,
            attention_engine=attention_engine,
        )

        self.blocks_in_memory = blocks_in_memory
        self.enable_block_swap = enable_block_swap
        self.num_visual_blocks = num_visual_blocks

        # Track which blocks are currently in GPU memory
        self._blocks_on_gpu = set()
        self._device = None

    def _ensure_block_on_gpu(self, block_idx, device):
        """Ensure a specific visual transformer block is on GPU"""
        if not self.enable_block_swap:
            return

        if block_idx in self._blocks_on_gpu:
            return

        # If we're at capacity, offload the oldest block
        if len(self._blocks_on_gpu) >= self.blocks_in_memory:
            # Offload the block that's furthest from current index
            # This is a simple FIFO-like strategy
            blocks_to_offload = sorted(self._blocks_on_gpu)[:1]
            for idx in blocks_to_offload:
                self.visual_transformer_blocks[idx].to('cpu', non_blocking=True)
                self._blocks_on_gpu.remove(idx)

        # Load the block to GPU
        self.visual_transformer_blocks[block_idx].to(device, non_blocking=True)
        self._blocks_on_gpu.add(block_idx)

    def _prefetch_blocks(self, start_idx, device, num_blocks=None):
        """Prefetch multiple blocks to GPU"""
        if num_blocks is None:
            num_blocks = self.blocks_in_memory

        for i in range(start_idx, min(start_idx + num_blocks, self.num_visual_blocks)):
            self._ensure_block_on_gpu(i, device)

    def offload_all_blocks(self):
        """Offload all visual transformer blocks from GPU to CPU"""
        if not self.enable_block_swap:
            return

        # Offload all blocks that are currently on GPU
        for idx in list(self._blocks_on_gpu):
            self.visual_transformer_blocks[idx].to('cpu', non_blocking=True)
        self._blocks_on_gpu.clear()

    def to(self, device, **kwargs):
        """Override to track device and initialize block swapping"""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self.enable_block_swap and device != 'cpu':
            # Move core components to GPU
            self.time_embeddings.to(device, **kwargs)
            self.text_embeddings.to(device, **kwargs)
            self.pooled_text_embeddings.to(device, **kwargs)
            self.visual_embeddings.to(device, **kwargs)
            self.text_rope_embeddings.to(device, **kwargs)
            self.visual_rope_embeddings.to(device, **kwargs)
            self.out_layer.to(device, **kwargs)

            # Move text transformer blocks (these are few, keep them all on GPU)
            for block in self.text_transformer_blocks:
                block.to(device, **kwargs)

            # For visual transformer blocks, prefetch first few blocks
            # Keep rest on CPU
            for i in range(self.num_visual_blocks):
                self.visual_transformer_blocks[i].to('cpu', **kwargs)

            self._blocks_on_gpu.clear()
            self._prefetch_blocks(0, device)

        else:
            # Normal behavior - move everything
            return super().to(device, **kwargs)

        return self

    def forward(
        self,
        x,
        text_embed,
        pooled_text_embed,
        time,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=(1.0, 1.0, 1.0),
        sparse_params=None,
        attention_mask=None
    ):
        """Forward pass with block swapping"""

        # Get device from input
        device = x.device

        # Text embedding and encoding (same as before)
        text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
            text_embed, time, pooled_text_embed, x, text_rope_pos)

        for text_transformer_block in self.text_transformer_blocks:
            text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)

        visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params)

        # Visual transformer blocks with block swapping
        if self.enable_block_swap:
            for i, visual_transformer_block in enumerate(self.visual_transformer_blocks):
                # Prefetch next block while processing current one
                if i + 1 < self.num_visual_blocks:
                    self._ensure_block_on_gpu(i + 1, device)

                # Ensure current block is on GPU
                self._ensure_block_on_gpu(i, device)

                # Process block
                visual_embed = visual_transformer_block(
                    visual_embed, text_embed, time_embed,
                    visual_rope, sparse_params, attention_mask
                )
        else:
            # Normal forward pass without swapping
            for visual_transformer_block in self.visual_transformer_blocks:
                visual_embed = visual_transformer_block(
                    visual_embed, text_embed, time_embed,
                    visual_rope, sparse_params, attention_mask
                )

        x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)
        return x


def get_dit_with_block_swap(conf, blocks_in_memory=4, enable_block_swap=False):
    """
    Create a DiT model with block swapping support.

    Args:
        conf: Model configuration
        blocks_in_memory: Number of visual transformer blocks to keep in GPU memory
        enable_block_swap: Whether to enable block swapping

    Returns:
        DiffusionTransformer3DBlockSwap instance
    """
    dit = DiffusionTransformer3DBlockSwap(
        blocks_in_memory=blocks_in_memory,
        enable_block_swap=enable_block_swap,
        **conf
    )
    return dit
