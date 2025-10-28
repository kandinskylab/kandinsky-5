import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    BitsAndBytesConfig
)

from .utils import freeze


class ClipTextEmbedder:
    def __init__(self, conf, device):
        self.model = CLIPTextModel.from_pretrained(conf.checkpoint_path).to(device)
        self.model = freeze(self.model)
        self.tokenizer = CLIPTokenizer.from_pretrained(conf.checkpoint_path)
        self.max_length = conf.max_length

    def __call__(self, texts):
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            pooled_embed = self.model(**inputs)["pooler_output"]
        return pooled_embed


class Qwen2_5_VLTextEmbedder:
    PROMPT_TEMPLATE = {
        "template": {
            "video": (
                "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe the location of the video, main characters or objects and their action.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ),
            "image": (
                "<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ),
        },
        "crop_start": {"video": 129, "image": 41},
    }

    def __init__(self, conf, device, quantized_qwen=False, text_token_padding=False):
        quantization_config = None
        if quantized_qwen:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            conf.checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=quantization_config
        )
        self.model = freeze(self.model)
        self.model = torch.compile(self.model, dynamic=True)
        self.processor = AutoProcessor.from_pretrained(conf.checkpoint_path, use_fast=True)
        self.max_length = conf.max_length
        self.text_token_padding = text_token_padding

    def __call__(self, texts, type_of_content="video"):
        prompt_template = "\n".join(self.PROMPT_TEMPLATE["template"][type_of_content])
        crop_start = self.PROMPT_TEMPLATE["crop_start"][type_of_content]
        full_texts = list(map(lambda x: prompt_template.format(x), texts))

        max_length = self.max_length + crop_start
        inputs = self.processor(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(self.model.device)

        with torch.no_grad():
            embeds = self.model(
                input_ids=inputs["input_ids"],
                return_dict=True,
                output_hidden_states=True,
            )["hidden_states"][-1][:, crop_start:]
        attention_mask = inputs["attention_mask"][:, crop_start:]
        if self.text_token_padding:
            seq_length = embeds.shape[1]
            cu_seqlens = torch.tensor([0, seq_length], dtype=torch.int32)
        else:
            embeds = embeds[attention_mask.bool()]
            cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
            cu_seqlens = torch.cat([torch.zeros_like(cu_seqlens)[:1], cu_seqlens]).to(
                dtype=torch.int32
            )
            attention_mask = None
        return embeds, cu_seqlens, attention_mask



class Kandinsky5TextEmbedder:
    def __init__(self, conf, device="cpu", quantized_qwen=False, text_token_padding=False):
        self.embedder = Qwen2_5_VLTextEmbedder(conf.qwen, device, quantized_qwen, text_token_padding)
        self.clip_embedder = ClipTextEmbedder(conf.clip, device)
        self.conf = conf

    def encode(self, texts, type_of_content="image"):
        text_embeds, cu_seqlens, attention_mask = self.embedder(texts, type_of_content=type_of_content)
        pooled_embed = self.clip_embedder(texts)
        return {"text_embeds": text_embeds, "pooled_embed": pooled_embed}, cu_seqlens, attention_mask

    def to(self, device):
        self.embedder.model = self.embedder.model.to(device)
        self.clip_embedder.model = self.clip_embedder.model.to(device)
        return self


def get_text_embedder(conf, device="cpu", quantized_qwen=False, text_token_padding=False):
    return Kandinsky5TextEmbedder(conf, device, quantized_qwen, text_token_padding)
