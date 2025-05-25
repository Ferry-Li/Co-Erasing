import os
import random
from tqdm import tqdm
import torch
import yaml
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.model_utils import load_unet, load_others
from utils.training_utils import get_training_params
from utils.diffusion_utils import denoise_to_text_timestep, predict_text_t_noise, set_scheduler_device
from utils.helpers import save_model, to_same_device


def train_text_mode(args):
    device_list = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    device_1 = torch.device(device_list[0])
    device_2 = torch.device(device_list[1])

    origin_unet = load_unet(args.ckpt_path, requires_grad=False).to(device_1)
    unet = load_unet(args.ckpt_path, requires_grad=True).to(device_2)
    

    if args.unet_ckpt_path:
        unet.load_state_dict(torch.load(args.unet_ckpt_path))
        origin_unet.load_state_dict(torch.load(args.unet_ckpt_path))

    vae, tokenizer, text_encoder, noise_scheduler, _ = load_others(args.ckpt_path, requires_grad=False)
    text_encoder = text_encoder.to(origin_unet.device)

    unet.train()
    origin_unet.eval()

    optimizer = torch.optim.Adam(get_training_params(unet, args.train_method), lr=args.lr)
    criterion = torch.nn.MSELoss()
    num_inference_steps = args.num_inference_steps

    noise_scheduler.set_timesteps(num_inference_steps)

    save_path = args.save_path or os.path.join("checkpoints", args.modality, args.prompt)
    unet_save_path = os.path.join(save_path, "unet")
    os.makedirs(unet_save_path, exist_ok=True)

    writer = SummaryWriter(unet_save_path)
    prompt_list = [p.strip() for p in args.prompt.split(',')]

    for idx in tqdm(range(args.iterations)):
        optimizer.zero_grad()

        prompt = random.choice(prompt_list)
        input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids
        text_embed = text_encoder(input_ids.to(origin_unet.device))[0].to(unet.device)

        uncond_ids = tokenizer("", return_tensors="pt", padding="max_length", truncation=True).input_ids
        uncond_embed = text_encoder(uncond_ids.to(origin_unet.device))[0].to(unet.device)

        t = torch.randint(num_inference_steps, (1,)).to(unet.device)
        t_ddpm = torch.randint(int(t * 1000 / num_inference_steps),
                               int((t + 1) * 1000 / num_inference_steps), (1,))

        start_code = torch.randn((1, 4, 64, 64)).to(unet.device)

        with torch.no_grad():
            set_scheduler_device(noise_scheduler, unet.device)
            z = denoise_to_text_timestep(unet, text_embed, t, start_code, noise_scheduler)
            uncond_origin_noise = predict_text_t_noise(z, t_ddpm, origin_unet, uncond_embed)
            cond_origin_noise = predict_text_t_noise(z, t_ddpm, origin_unet, text_embed)

        cond_noise = predict_text_t_noise(z, t_ddpm, unet, text_embed)

        [cond_origin_noise, uncond_origin_noise, cond_noise] = to_same_device(
            [cond_origin_noise, uncond_origin_noise, cond_noise], unet.device)

        loss = criterion(cond_noise, uncond_origin_noise - args.negative_guidance * (cond_origin_noise - uncond_origin_noise))
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), idx)

        if (idx + 1) % args.save_iter == 0:
            save_model(unet, unet_save_path, idx)
            print(f"[Checkpoint] Saved at iteration {idx + 1}")
