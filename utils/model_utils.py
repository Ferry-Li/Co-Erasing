import torch
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.unets import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.schedulers import DDPMScheduler
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        tokens = self.proj(image_embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        return self.norm(tokens)


class IPAdapter(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        if ckpt_path:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, schedule=None):
        if image_embeds.shape[-1] != 768:
            ip_tokens = self.image_proj_model(image_embeds)
        else:
            ip_tokens = image_embeds

        if schedule is not None:
            schedule = schedule.to(self.unet.device)
            encoder_hidden_states *= schedule
            ip_tokens *= (1 - schedule)

        hidden = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        return self.unet(noisy_latents, timesteps, hidden).sample

    def load_from_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"])
        print(f"[IPAdapter] Loaded checkpoint: {ckpt_path}")


def get_attn_processor(unet):
# init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    return attn_procs


def load_unet(ckpt_path, requires_grad=False):
    unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")
    unet.requires_grad_(requires_grad)
    return unet


def load_others(ckpt_path, requires_grad=False, image_encoder_path=None):
    vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder")
    scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path) if image_encoder_path else None

    # Freeze all
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if image_encoder:
        image_encoder.requires_grad_(False)

    return vae, tokenizer, text_encoder, scheduler, image_encoder
