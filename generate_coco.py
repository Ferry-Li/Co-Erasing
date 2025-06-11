import argparse
import os
import torch
import torch.utils.checkpoint
from tqdm.auto import tqdm
from diffusers import AutoPipelineForText2Image
import pandas as pd
from transformers import CLIPTextModel

prompt_path = "prompts/coco_10k.csv"
CLEAN_SD_PATH = "" # set the clean stable diffusion path here

pipeline = AutoPipelineForText2Image.from_pretrained(CLEAN_SD_PATH, 
                                                     # torch_dtype=torch.float16,
                                                     use_safetensors=True,
                                                     local_files_only=True,
                                                     safety_checker=None).to("cuda")


parser = argparse.ArgumentParser(
                    prog = 'generating COCO...',
                    description = 'generating COCO...')
    
parser.add_argument('--path', help='path to the erased model (unet)', type=str, required=True, default=None)
args = parser.parse_args()

pipeline.unet.load_state_dict(torch.load(args.path, map_location='cuda'), strict=False)
pipeline.set_progress_bar_config(disable=True)
save_name = "" # set the saving dir here
save_dir = f'results/coco-10k/{save_name}'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

prompt_data = pd.read_csv(prompt_path)

for index, row in tqdm(prompt_data.iterrows()):
    prompt = row["prompt"]
    seed = row["evaluation_seed"]
    case_number = row["case_number"]
    generator = torch.manual_seed(seed)
    image = pipeline(prompt=prompt, generator=generator).images[0]
    image_path = os.path.join(save_dir, f"{case_number}.png")  # Save images with zero-padded numbering
    image.save(image_path, "PNG")


