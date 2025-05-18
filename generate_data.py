import argparse
import os
from contextlib import nullcontext
from pathlib import Path
import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
from tqdm.auto import tqdm
from diffusers import AutoPipelineForText2Image
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--label", type=str, default=None)

args = parser.parse_args()

assert args.label is not None, "set a label"

pipeline = AutoPipelineForText2Image.from_pretrained("/data/feiran/stable-diffusion-v1-5", 
                                                     torch_dtype=torch.float16,
                                                     use_safetensors=True,
                                                     local_files_only=True,
                                                     safety_checker=None).to("cuda")

root_dir = "/data/feiran/robustDiffusion/generation_dataset_v1_5"
if not os.path.exists(root_dir):
    os.mkdir(root_dir)
label_dict = [args.label]
numbers_per_class = 500

for label in label_dict:
    save_dir = os.path.join(root_dir, label)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if label in ['Picasso', 'Van Gogh']:
        prompt = f'a drawing of {label}'
    else:
        prompt = f"a photo of {label}"

    for idx in tqdm(range(numbers_per_class), desc=f"Generating images for {label}"):
        image = pipeline(prompt=prompt).images[0]
        image_path = os.path.join(save_dir, f"{label}_{idx:04d}.png")  # Save images with zero-padded numbering
        image.save(image_path, "PNG")
        

# /data/feiran/robustDiffusion/generation_dataset/nudity/nudity_[xxxx].png: ['0000', '0005', ]