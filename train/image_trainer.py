import os
import random
import yaml
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from utils.training_utils import get_training_params
from utils.model_utils import (
    load_unet, load_others,
    ImageProjModel, IPAdapter, get_attn_processor
)
from utils.diffusion_utils import (
    set_scheduler_device, denoise_to_text_timestep,
    predict_image_t_noise, predict_text_t_noise
)
from utils.helpers import to_same_device, save_model


transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def train_image_mode(args):
    # Device setup
    device_list = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    device_1 = torch.device(device_list[0])
    device_2 = torch.device(device_list[1])

    # Load models
    origin_unet = load_unet(args.ckpt_path, requires_grad=False).to(device_1)
    unet = load_unet(args.ckpt_path, requires_grad=True).to(device_2)
    

    if args.unet_ckpt_path:
        unet.load_state_dict(torch.load(args.unet_ckpt_path))
        origin_unet.load_state_dict(torch.load(args.unet_ckpt_path))

    unet.train()
    origin_unet.eval()

    vae, tokenizer, text_encoder, noise_scheduler, _ = load_others(args.ckpt_path, requires_grad=False)
    text_encoder = text_encoder.to(origin_unet.device)

    # Init optimizer
    parameters = get_training_params(unet, args.train_method)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    criterion = torch.nn.MSELoss()
    num_inference_steps = args.num_inference_steps

    noise_scheduler.set_timesteps(num_inference_steps)

    # Define save path
    save_dir = args.save_path or os.path.join("checkpoints", args.modality, args.prompt)
    unet_save_path = os.path.join(save_dir, f"unet_{args.train_method}_im{args.image_number}_ng{args.negative_guidance}_it{args.iterations}")
    os.makedirs(unet_save_path, exist_ok=True)

    writer = SummaryWriter(unet_save_path)

    # Image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).to(origin_unet.device)
    image_encoder.requires_grad_(False)

    # Image projection models
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    origin_image_proj_model = ImageProjModel(
        cross_attention_dim=origin_unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    # Attention processors
    attn_procs = get_attn_processor(unet)
    unet.set_attn_processor(attn_procs)

    origin_attn_procs = get_attn_processor(origin_unet)
    origin_unet.set_attn_processor(origin_attn_procs)

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    origin_adapter_modules = torch.nn.ModuleList(origin_unet.attn_processors.values())
    ip_adapter_path = args.ip_adapter

    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, ip_adapter_path).to(unet.device)
    origin_ip_adapter = IPAdapter(origin_unet, origin_image_proj_model, origin_adapter_modules, ip_adapter_path).to(origin_unet.device)


    # Prepare text embeddings
    text_input_ids = tokenizer(args.prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids
    text_embeddings = text_encoder(text_input_ids.to(origin_unet.device))[0].to(unet.device)

    uncond_input_ids = tokenizer("", return_tensors="pt", padding="max_length", truncation=True).input_ids
    uncond_text_embeddings = text_encoder(uncond_input_ids.to(origin_unet.device))[0].to(unet.device)

    # Collect image embeddings (direct or contrastive)
    use_direct_image = args.image is not None
    use_contrastive = args.contrastive_image is not None
    use_text_guide = args.text_guide is not None
    assert use_direct_image or use_contrastive, "Set at least one image guidance method!"

    if use_direct_image:
        image_list = _load_image_list(args.image, args.image_number)
        print(f"[INFO] Found {len(image_list)} images to erase from: {args.image}")

    if use_contrastive:
        concept_embeds = _collect_contrastive_embeddings(args, image_encoder)
        concept_embed_tensor = torch.mean(torch.stack(concept_embeds), dim=0)

    # Training loop
    for idx in tqdm(range(args.iterations), desc="[Image Training]"):
        optimizer.zero_grad()

        # Sample image embedding
        if use_direct_image:
            image_embeds = _sample_image_embedding(image_list, image_encoder)

        elif use_contrastive:
            image_embeds = _sample_contrastive_embedding(args, concept_embeds, concept_embed_tensor)

        if use_text_guide:
            key = text_embeddings
            query = origin_ip_adapter.image_proj_model(image_embeds)
            value = key
            
            attention_scores = torch.matmul(query.to(origin_unet.device), key.to(origin_unet.device).transpose(1, 2))
                # Scale the attention scores
            d_k = key.size(-1)  # embedding_dim
            scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
            
            # Apply softmax to get the attention weights
            attention_weights = F.softmax(scaled_attention_scores, dim=-1)  # Shape: [batch_size, 1, num_patches]
            
            # Compute the weighted sum of the image embeddings (weighted by attention)
            attended_image_embedding = torch.matmul(attention_weights, value.to(origin_unet.device))  # Shape: [batch_size, 1, embedding_dim]
            image_embeds = attended_image_embedding

        # Add noise to image embedding if specified
        if args.noise_factor > 0:
            noise = torch.rand_like(image_embeds)
            image_embeds = image_embeds + args.noise_factor * noise

        # Sample timestep
        t = torch.randint(num_inference_steps, (1,)).to(unet.device)
        t_ddpm = torch.randint(int(t * 1000 / num_inference_steps), int((t + 1) * 1000 / num_inference_steps), (1,))

        # Start from noise
        start_code = torch.randn((1, 4, 64, 64)).to(unet.device)

        with torch.no_grad():
            set_scheduler_device(noise_scheduler, unet.device)
            z = denoise_to_text_timestep(unet, text_embeddings, t, start_code, noise_scheduler)
            if args.blur_factor > 0:
                z = torchvision.transforms.functional.gaussian_blur(z, kernel_size=args.blur_factor)

            cond_origin_noise = predict_image_t_noise(z, t_ddpm, origin_unet, text_embeddings, origin_ip_adapter, image_embeds)
            if args.text_uncond:
                uncond_origin_noise = predict_text_t_noise(z, t_ddpm, origin_unet, uncond_text_embeddings)
            else:
                uncond_origin_noise = predict_image_t_noise(z, t_ddpm, origin_unet, uncond_text_embeddings, origin_ip_adapter, image_embeds)

        cond_noise = predict_image_t_noise(z, t_ddpm, unet, text_embeddings, ip_adapter, image_embeds)

        # Compute loss
        cond_noise, uncond_origin_noise, cond_origin_noise = to_same_device(
            [cond_noise, uncond_origin_noise, cond_origin_noise], unet.device)

        loss = criterion(cond_noise, uncond_origin_noise - args.negative_guidance * (cond_origin_noise - uncond_origin_noise))

        loss.backward()
        optimizer.step()

        writer.add_scalar('image_training_loss', loss.item(), idx + 1)

        if (idx + 1) % args.save_iter == 0:
            save_model(unet, unet_save_path, idx)
            print(f"[Checkpoint] Saved UNet at iteration {idx + 1}")


# ========================
#  Support functions
# ========================

def _load_image_list(image_path, image_number):
    if os.path.isdir(image_path):
        all_images = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(('png', 'jpg'))]
        return random.sample(all_images, min(image_number, len(all_images)))
    else:
        return [image_path]


def _sample_image_embedding(image_list, image_encoder):
    image_path = random.choice(image_list)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(image_encoder.device)
    return image_encoder(image_tensor).image_embeds


def _collect_contrastive_embeddings(args, image_encoder):
    concept_embeds = []
    pair_dirs = [os.path.join(args.contrastive_image, d) for d in os.listdir(args.contrastive_image)
                 if os.path.isdir(os.path.join(args.contrastive_image, d))]

    for pair_dir in tqdm(pair_dirs, desc="[Contrastive]"):
        pos = transform(Image.open(os.path.join(pair_dir, "positive.png")).convert("RGB")).unsqueeze(0).to(image_encoder.device)
        neg = transform(Image.open(os.path.join(pair_dir, "negative.png")).convert("RGB")).unsqueeze(0).to(image_encoder.device)

        pos_embed = image_encoder(pos).image_embeds
        neg_embed = image_encoder(neg).image_embeds

        if args.diff_method == "diff":
            embed = pos_embed - neg_embed
        elif args.diff_method == "l1":
            embed = torch.abs(pos_embed - neg_embed)
        elif args.diff_method == "l2":
            embed = (pos_embed - neg_embed) ** 2
        else:
            raise ValueError(f"Invalid diff_method: {args.diff_method}")

        concept_embeds.append(embed)

    return concept_embeds


def _sample_contrastive_embedding(args, concept_embeds, avg_embed):
    if args.use_average:
        return avg_embed
    else:
        return random.choice(concept_embeds)
