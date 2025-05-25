import os
import argparse
import torch


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modality', type=str, choices=['text', 'image'], default='text')
    parser.add_argument('--train_method', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='/data/feiran/stable-diffusion-v1-5')
    parser.add_argument('--unet_ckpt_path', type=str, default=None)
    parser.add_argument('--image_encoder_path', type=str, default='/data/feiran/stable-diffusion-v1-5/image_encoder/')
    parser.add_argument('--ip_adapter', type=str, default='/data/feiran/stable-diffusion-v1-5/ip_adapter/ip-adapter_sd15.bin')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--save_iter', type=int, default=500)
    parser.add_argument('--negative_guidance', type=float, default=1.0)
    parser.add_argument('--image', type=str, default='path/to/images')
    parser.add_argument('--contrastive_image', type=str, default=None)
    parser.add_argument('--image_number', type=int, default=100)
    parser.add_argument('--noise_factor', type=float, default=0.0)
    parser.add_argument('--blur_factor', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--text_uncond", action='store_true')
    parser.add_argument("--text_guide", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="log")
    parser.add_argument('--num_inference_steps', type=int, default=50)
    return parser.parse_args()


def save_model(model, save_path, idx=-1):
    os.makedirs(save_path, exist_ok=True)
    state_dict = model.state_dict().copy()
    keys_to_remove = ['to_k_ip.weight', 'to_v_ip.adapter']
    for key in list(state_dict.keys()):
        if any(sub in key for sub in keys_to_remove):
            del state_dict[key]
    file = f"unet_{idx}.pth" if idx != -1 else "unet.pth"
    torch.save(state_dict, os.path.join(save_path, file))


def to_same_device(tensors, device):
    return [t.to(device) for t in tensors]
