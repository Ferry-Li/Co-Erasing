from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
import random
import os
import numpy as np
from glob import glob
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import yaml
import argparse
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPProcessor, CLIPModel
from utils import IPAdapter, ImageProjModel
from utils import get_training_params, get_attn_processor
from utils import load_unet, load_others, anneal_schedule
from utils import estimate_x0, forward_diffusion_sample


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out



def init_args():
    parser = argparse.ArgumentParser(prog = 'Text and Image Erase',
                    description = 'Finetuning stable diffusion model to erase concepts using text and image cross attention')
    parser.add_argument('--modality', type=str, default='text', help='[text, image]')
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--train_method', help='method of training: [noxattn, selfattn, xattn, full]', type=str, required=True)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--image_number', help='number of images used in erasing', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
    parser.add_argument('--image', help='path to erasing images', type=str, default=None)
    parser.add_argument('--neutral_image', help='path to normal images', type=str)
    parser.add_argument('--image_embedding', help='path to erasing image embeddings', type=str)
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str)
    parser.add_argument('--clip_path', type=str, default='/data/feiran/sd-xl-base-1.0/openai/clip-vit-large-patch14')
    parser.add_argument('--image_encoder_path', help='path to the image encoder', type=str, default='/data/feiran/stable-diffusion-v1-5/image_encoder/')
    parser.add_argument('--ip_adapter', help='path to the ip-adapter', type=str, default='/data/feiran/stable-diffusion-v1-5/ip_adapter/ip-adapter_sd15.bin')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion', type=str, required=False, default='/data/feiran/stable-diffusion-v1-5')
    parser.add_argument('--unet_ckpt_path', help='ckpt path for stable diffusion', type=str, required=False, default=None)
    # /data/feiran/robustDiffusion/MultiErase/checkpoints/text/nudity/unet.pth
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='1,2,4,5')
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--logging_dir", type=str, default="log")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--save_iter", type=int, default=500)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--use_schedule", action='store_true')
    parser.add_argument("--negative_guidance", type=float, default=1.0)
    parser.add_argument("--noise_factor", type=float, default=0)
    parser.add_argument("--blur_factor", type=int, default=0)
    parser.add_argument("--clip_reg", type=float, default=0)
    parser.add_argument("--similarity_reg", type=float, default=0)
    parser.add_argument("--contrastive_image", type=str, default=None)
    parser.add_argument("--use_average", action='store_true')
    parser.add_argument("--use_anchor", action='store_true')
    parser.add_argument("--diff_method", type=str, default="diff")
    parser.add_argument("--image_uncond", action='store_true')
    parser.add_argument("--text_uncond", action='store_true')
    parser.add_argument("--contrastive_size", type=int, default=1)
    parser.add_argument("--text_guide", type=str, default=None)
    
    args = parser.parse_args()

    return args

'''
image: 0004, 0005, 0011, 0013, 0016, 0026, 0035, 0072, 
'''


def save_model(model, save_path, idx=-1):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    state_dict = model.state_dict().copy()

    # Define the keys to be removed
    keys_to_remove = ['to_k_ip.weight', 'to_v_ip.adapter']

    # Remove the specified keys from the copied state dictionary
    for key in list(state_dict.keys()):
        if any(substring in key for substring in keys_to_remove):
            del state_dict[key]
    if idx != -1:
        torch.save(model.state_dict(), os.path.join(save_path, f"unet_{idx}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, "unet.pth"))


transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])



def denoise_to_text_timestep(unet, text_embeddings, t, start_code, noise_scheduler):

    latents = start_code
    # Step 3: Get the total number of timesteps from the scheduler
    num_timesteps = noise_scheduler.num_inference_steps
    timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=num_timesteps, num_ddpm_timesteps=1000, verbose=False)
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]

    # Step 4: Iteratively denoise down to timestep `t`
    for i, step in enumerate(time_range):
        # step: [990 975 ...] len(step) = num_inference_steps
        index = total_steps - i - 1
        t_tensor = torch.tensor([step], dtype=torch.long).to(unet.device)

        # Predict the noise
        with torch.no_grad():
            noise_pred = unet(latents, t_tensor, encoder_hidden_states=text_embeddings).sample

        # Get the denoised latent
        latents = noise_scheduler.step(noise_pred, t_tensor, latents).prev_sample.to(unet.device)
        
        if index == t:
            break

    denoised_image = latents
    return denoised_image

def denoise_to_image_timestep(unet, text_embeddings, t, start_code, noise_scheduler, ip_adapter, image_embeds, schedule=None):
    
    latents = start_code
    # Step 3: Get the total number of timesteps from the scheduler
    num_timesteps = noise_scheduler.num_inference_steps
    timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=num_timesteps, num_ddpm_timesteps=1000, verbose=False)
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]

    # Step 4: Iteratively denoise down to timestep `t`
    for i, step in enumerate(time_range):
        # step: [990 975 ...] len(step) = num_inference_steps
        index = total_steps - i - 1
        t_tensor = torch.tensor([step], dtype=torch.long).to(unet.device)

        # Predict the noise
        with torch.no_grad():
            noise_pred = ip_adapter(latents, t_tensor, text_embeddings.to(unet.device), image_embeds.to(unet.device), schedule)

        # Get the denoised latent
        latents = noise_scheduler.step(noise_pred, t_tensor, latents).prev_sample.to(unet.device)

        if index == t:
            break

    # Step 5: Rescale the final latent to [0, 1] range for visualization
    denoised_image = latents

    return denoised_image

def predict_text_t_noise_simple(z, t_ddpm, unet, text_embeddings):
    noise_pred = unet(
                z.to(unet.device),
                t_ddpm.to(unet.device),
                encoder_hidden_states=text_embeddings.to(unet.device),
                return_dict=False,
            )[0]
    return noise_pred

def predict_image_t_noise_simple(z, t_ddpm, unet, text_embedding, ip_adapter, image_embeds, schedule=None):
    noise_pred = ip_adapter(z.to(unet.device), t_ddpm.to(unet.device), text_embedding.to(unet.device), image_embeds.to(unet.device), schedule)

    return noise_pred


def to_same_device(item_list, device):
    for idx in range(len(item_list)):
        item_list[idx] = item_list[idx].to(device)
    return item_list

def unlearning(args):

    train_method = args.train_method
    device_list = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    train_method = args.train_method
    device_1 = torch.device(device_list[0])
    unet_device_1 = torch.device(device_list[0])
    unet_device_2 = torch.device(device_list[1])

    # 1. load Stable Diffusion pipelines

    # origin_pipeline:  referenced model
    # pipeline:         trained model

    origin_unet = load_unet(args.ckpt_path, requires_grad=False).to(unet_device_1)
    if args.unet_ckpt_path is not None:
        origin_unet.load_state_dict(torch.load(args.unet_ckpt_path))
        print(f"loading unet from {args.unet_ckpt_path}")
    origin_unet.eval()
 
    vae, tokenizer, text_encoder, noise_scheduler, image_encoder = load_others(args.ckpt_path, requires_grad=False, image_encoder_path=args.image_encoder_path)
    vae, image_encoder = vae.to(device_1), image_encoder.to(device_1)
    text_encoder = text_encoder.to(device_1)

    # Finetuning!
    if args.modality == 'text':

        unet = load_unet(args.ckpt_path, requires_grad=True).to(unet_device_2)
        if args.unet_ckpt_path is not None:
            unet.load_state_dict(torch.load(args.unet_ckpt_path))
        unet.train()
        # 2. Choose parameters to train based on train_method
        parameters = get_training_params(unet, train_method)
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        criterion = torch.nn.MSELoss()


        with open('configs/text_erase.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # 3.1 text-based erasing
        text_prompt_list = args.prompt.split(',')
        text_prompt_list = [text.strip() for text in text_prompt_list]

        print(f"Erasing {text_prompt_list}...")

        # 0. Default height and width to unet
        negative_guidance = args.negative_guidance
        noise_scheduler.set_timesteps(num_inference_steps=config["num_inference_steps"])
        if args.save_path is None:
            if args.use_schedule:
                save_path = os.path.join("checkpoints", args.modality, args.prompt + '_schedule')
            else:
                save_path = os.path.join("checkpoints", args.modality, args.prompt)
        else:
            save_path = args.save_path
            
        unet_save_path = os.path.join(save_path, "unet")
        if not os.path.exists(unet_save_path):
            os.makedirs(unet_save_path)
        
        log_dir = unet_save_path
        writer = SummaryWriter(log_dir)

        with open(os.path.join(log_dir, "log.txt"), "w") as f:
            f.write("Arguments:\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write("\nConfig:\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        

        # training loop
        training_bar = tqdm(range(args.iterations))
        running_loss = 0.0
        
        for idx in training_bar:
            optimizer.zero_grad()
            text_prompt = random.sample(text_prompt_list, 1)[0]
            text_input_ids = tokenizer(text_prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
            text_embeddings = text_encoder(text_input_ids.to(device_1), output_hidden_states=True)[0].to(unet.device)

            uncond_text_prompt = ""
            uncond_text_input_ids = tokenizer(uncond_text_prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
            uncond_text_embeddings = text_encoder(uncond_text_input_ids.to(device_1), output_hidden_states=True)[0].to(unet.device)


            num_inference_steps = config["num_inference_steps"]
            t = torch.randint(num_inference_steps, (1,)).to(unet.device)

            og_num = round((int(t)/num_inference_steps)*1000)
            og_num_lim = round((int(t+1)/num_inference_steps)*1000)
            t_ddpm = torch.randint(og_num, og_num_lim, (1,))
            
            start_code = torch.randn((1, 4, 64, 64)).to(unet.device)
            
            with torch.no_grad():
                noise_scheduler.set_device(unet.device)
                z = denoise_to_text_timestep(unet, text_embeddings, t, start_code, noise_scheduler)
                uncond_origin_noise = predict_text_t_noise_simple(z, t_ddpm, origin_unet, uncond_text_embeddings)
                cond_origin_noise = predict_text_t_noise_simple(z, t_ddpm, origin_unet, text_embeddings)
            cond_noise = predict_text_t_noise_simple(z, t_ddpm, unet, text_embeddings)

            uncond_origin_noise.requires_grad = False
            cond_origin_noise.requires_grad = False

            [cond_origin_noise, uncond_origin_noise, cond_noise] = to_same_device([cond_origin_noise, uncond_origin_noise, cond_noise], unet.device)
            loss = criterion(cond_noise, uncond_origin_noise - (negative_guidance * (cond_origin_noise - uncond_origin_noise)))
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar('text_training_loss', loss.item(), idx + 1)

            if (idx + 1) % args.save_iter == 0:
                save_path = unet_save_path
                save_model(unet, save_path)
                print(f"save to {save_path} at iter:{idx + 1}")

    elif args.modality == 'image':

        use_direct_image = (args.image is not None)
        use_contrastive_feature = (args.contrastive_image is not None)
        use_text_guide = args.text_guide is not None

        assert use_direct_image or use_contrastive_feature, "Set at least one image guidance!"

        unet = load_unet(args.ckpt_path, requires_grad=True).to(unet_device_2)
        if args.unet_ckpt_path is not None:
            unet.load_state_dict(torch.load(args.unet_ckpt_path))
            print(f"loading unet from {args.unet_ckpt_path}")
        unet.train()
        # 2. Choose parameters to train based on train_method
        parameters = get_training_params(unet, train_method)
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        criterion = torch.nn.MSELoss()


        with open('configs/text_erase.yaml', 'r') as f:
            config = yaml.safe_load(f)

        negative_guidance = args.negative_guidance
        noise_scheduler.set_timesteps(config["num_inference_steps"])

        if args.save_path is None:
            if args.use_schedule:
                save_path = os.path.join("checkpoints", args.modality, args.prompt + '_schedule')
            else:
                save_path = os.path.join("checkpoints", args.modality, args.prompt)
        else:
            save_path = args.save_path

        unet_save_path = os.path.join(save_path, f"unet_{args.train_method}_im{args.image_number}_ng{negative_guidance}_it{args.iterations}")
        if args.similarity_reg != 0:
            unet_save_path = unet_save_path + f"_sr{args.similarity_reg}"
        if args.clip_reg != 0:
            unet_save_path = unet_save_path + f"_sr{args.clip_reg}"
        if use_contrastive_feature:
            unet_save_path = unet_save_path + f"_contrastive_{args.diff_method}"
        elif use_direct_image:
            unet_save_path = unet_save_path + f"_direct"
        if not args.use_average:
            unet_save_path = unet_save_path + f"_random"
        if args.contrastive_size != 1:
            unet_save_path = unet_save_path + f"_contraSize{args.contrastive_size}"
        if args.use_anchor:
            unet_save_path = unet_save_path + f"_anchor"
        if args.image_uncond:
            unet_save_path = unet_save_path + f"_imuncond"
        elif args.text_uncond:
            unet_save_path = unet_save_path + f"_txuncond"
        if args.text_guide is not None:
            unet_save_path = unet_save_path + f"_textGuide-{args.text_guide}"
        if args.noise_factor != 0:
            unet_save_path = unet_save_path + f"_nf{args.noise_factor}"
        if args.blur_factor != 0:
            unet_save_path = unet_save_path + f"_bf{args.blur_factor}"
        if args.unet_ckpt_path is None:
            unet_save_path = unet_save_path + f"_NoPre"
        


        if not os.path.exists(unet_save_path):
            os.makedirs(unet_save_path)
        
        log_dir = unet_save_path

        writer = SummaryWriter(log_dir)
        

        with open(os.path.join(log_dir, "log.txt"), "w") as f:
            f.write("Arguments:\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write("\nConfig:\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")


        # This unet in the new pipeline is to be finetuned
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).to(device_1)
        # origin_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
        image_encoder.requires_grad_(False)
        # origin_image_encoder.requires_grad_(False)
        
        # ip_adapters
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

        attn_procs = get_attn_processor(unet)
        unet.set_attn_processor(attn_procs)

        origin_attn_procs = get_attn_processor(origin_unet)
        origin_unet.set_attn_processor(origin_attn_procs)

        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        origin_adapter_modules = torch.nn.ModuleList(origin_unet.attn_processors.values())
        ip_adapter_path = args.ip_adapter
        ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, ip_adapter_path).to(unet.device)
        origin_ip_adapter = IPAdapter(origin_unet, origin_image_proj_model, origin_adapter_modules, ip_adapter_path).to(origin_unet.device)

        # 3.1 image-based erasing
        text_prompt = args.prompt
        text_ipnut_ids = tokenizer(text_prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        text_embeddings = text_encoder(text_ipnut_ids.to(device_1), output_hidden_states=True)[0].to(unet.device)

        uncond_prompt = ""
        uncond_text_ipnut_ids = tokenizer(uncond_prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        uncond_text_embeddings = text_encoder(uncond_text_ipnut_ids.to(device_1), output_hidden_states=True)[0].to(unet.device)

        num_inference_steps = config["num_inference_steps"]
        t = torch.randint(num_inference_steps, (1,)).to(unet.device)

        og_num = round((int(t)/num_inference_steps)*1000)
        og_num_lim = round((int(t+1)/num_inference_steps)*1000)
        t_ddpm = torch.randint(og_num, og_num_lim, (1,))

        # neutral_imae: neutral images or anchor points?
        if not args.use_anchor:
            if args.image_uncond:
                neutral_image_list = [os.path.abspath(os.path.join(args.neutral_image, file)) for file in os.listdir(args.neutral_image) if os.path.isfile(os.path.join(args.neutral_image, file))]
            else:
                neutral_image_list = []
        else:
            neutral_image_list = []
            pair_list = [os.path.abspath(os.path.join(args.contrastive_image, file)) for file in os.listdir(args.contrastive_image) if os.path.isdir(os.path.join(args.contrastive_image, file))]

            for pair in tqdm(pair_list):
                neutral_image_list.append(os.path.join(pair, "neutral.png"))
                


        if use_direct_image:
            if os.path.isdir(args.image):
                image_list = [os.path.abspath(os.path.join(args.image, file)) for file in os.listdir(args.image) if os.path.isfile(os.path.join(args.image, file))]
                if args.image_number <= len(image_list):
                    image_list = random.sample(image_list, args.image_number)
                    if len(neutral_image_list) > 0:
                        neutral_image_list = random.sample(neutral_image_list, args.image_number)
            else:
                image_list = [args.image]
            print(f"Erasing images from {args.image} ...")
        
        elif use_contrastive_feature:
            pair_list = [os.path.abspath(os.path.join(args.contrastive_image, file)) for file in os.listdir(args.contrastive_image) if os.path.isdir(os.path.join(args.contrastive_image, file))]
            if args.image_number <= len(pair_list):
                pair_list = random.sample(pair_list, args.image_number)
                neutral_image_list = random.sample(neutral_image_list, args.image_number)
            print(f"Erasing pair-images from {args.contrastive_image} ...")

            concept_embeds = []
            positive_embds = []
            negative_embeds = []
            for pair in tqdm(pair_list):
                positive_image = Image.open(os.path.join(pair, "positive.png"))
                positive_image = transform(positive_image.convert("RGB")).to(image_encoder.device)
                positive_image = positive_image.unsqueeze(0)
                positive_image_embeds = image_encoder(positive_image).image_embeds
                positive_embds.append(positive_image_embeds)

                negative_image = Image.open(os.path.join(pair, "negative.png"))
                negative_image = transform(negative_image.convert("RGB")).to(image_encoder.device)
                negative_image = negative_image.unsqueeze(0)
                negative_image_embeds = image_encoder(negative_image).image_embeds
                negative_embeds.append(negative_image_embeds)
                
                # how to calculate difference
                if args.diff_method == 'diff':
                    concept_embed = positive_image_embeds - negative_image_embeds
                elif args.diff_method == 'l1':
                    concept_embed = torch.abs(positive_image_embeds - negative_image_embeds)
                elif args.diff_method == 'l2':
                    concept_embed = (positive_image_embeds - negative_image_embeds) ** 2
                
                concept_embeds.append(concept_embed)
            example_image = positive_image
            # random or average
            concept_embeds_tensor = torch.mean(torch.stack(concept_embeds), dim=0)

        # training loop
        training_bar = tqdm(range(args.iterations))

        for idx in training_bar:
            # seed = random.randint(0, int(1e6) - 1)
            # generator = torch.Generator()
            # generator.manual_seed(seed)
            optimizer.zero_grad()
            if use_direct_image:
                image = Image.open(random.sample(image_list, 1)[0])
                image = transform(image.convert("RGB")).to(image_encoder.device)
                image = image.unsqueeze(0)
                image_embeds = image_encoder(image).image_embeds
                noise = torch.rand_like(image)
                noise_embeds = image_encoder(noise).image_embeds

                if use_text_guide:
                    '''
                    query = text_embeddings # [1, 77, 768]
                    # image_embds:  [1, 1024]
                    key = origin_ip_adapter.image_proj_model(image_embeds) # [1, 4, 768]
                    value = key
                    '''
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
                    

            elif use_contrastive_feature:
                use_average = args.use_average
                if use_average:
                    image_embeds = concept_embeds_tensor
                else:
                    if args.contrastive_size == 1:
                        image_embeds = random.sample(concept_embeds, 1)[0]
                    else:
                        temp_positive_embeds = []
                        temp_negative_embeds = []
                        idx_list = random.sample(range(len(pair_list)), args.contrastive_size)
                        for idx in idx_list:
                            temp_positive_embeds.append(positive_embds[idx])
                            temp_negative_embeds.append(negative_embeds[idx])
                        temp_positive_embeds = torch.mean(torch.stack(temp_positive_embeds), dim=0)
                        temp_negative_embeds = torch.mean(torch.stack(temp_negative_embeds), dim=0)

                        if args.diff_method == 'diff':
                            image_embeds = temp_positive_embeds - temp_negative_embeds
                        elif args.diff_method == 'l1':
                            image_embeds = torch.abs(temp_positive_embeds - temp_negative_embeds)
                        elif args.diff_method == 'l2':
                            image_embeds = (temp_positive_embeds - temp_negative_embeds) ** 2


            if args.noise_factor != 0:
                noise = torch.rand_like(image_embeds)
                image_embeds = image_embeds + args.noise_factor * noise
            

            if args.image_uncond:
                neutral_image = Image.open(random.sample(neutral_image_list, 1)[0])
                neutral_image = transform(neutral_image.convert("RGB")).to(image_encoder.device).unsqueeze(0)
                neutral_image_embeds = image_encoder(neutral_image).image_embeds
                
                
            start_code = torch.randn((1, 4, 64, 64)).to(unet.device)

            with torch.no_grad():
                noise_scheduler.set_device(unet.device)
                if args.use_schedule:
                    schedule = anneal_schedule(idx, args.iterations)
                else:
                    schedule = None
                # z = denoise_to_image_timestep(unet, text_embeddings, t, start_code, noise_scheduler, ip_adapter, image_embeds, schedule)
                z = denoise_to_text_timestep(unet, text_embeddings, t, start_code, noise_scheduler)
                if args.blur_factor != 0:
                    z = torchvision.transforms.functional.gaussian_blur(z, kernel_size=args.blur_factor)
                breakpoint()
                if args.image_uncond:
                    uncond_origin_noise = predict_image_t_noise_simple(z, t_ddpm, origin_unet, uncond_text_embeddings, origin_ip_adapter, neutral_image_embeds, schedule)
                elif args.text_uncond:
                    uncond_origin_noise = predict_text_t_noise_simple(z, t_ddpm, origin_unet, uncond_text_embeddings)
                cond_origin_noise = predict_image_t_noise_simple(z, t_ddpm, origin_unet, text_embeddings, origin_ip_adapter, image_embeds, schedule)
            cond_noise = predict_image_t_noise_simple(z, t_ddpm, unet, text_embeddings, ip_adapter, image_embeds, schedule)

            
            [cond_origin_noise, uncond_origin_noise, cond_noise] = to_same_device([cond_origin_noise, uncond_origin_noise, cond_noise], unet.device)
            loss = criterion(cond_noise, uncond_origin_noise - (negative_guidance * (cond_origin_noise - uncond_origin_noise)))
            
            if args.similarity_reg != 0:
                uncond_noise = predict_image_t_noise_simple(z, t_ddpm, unet, uncond_text_embeddings, ip_adapter, neutral_image_embeds, schedule)
                similarity_reg_loss = criterion(uncond_noise, cond_noise)
                loss = loss + args.similarity_reg * similarity_reg_loss

            if args.clip_reg != 0:
                clip_model = CLIPModel.from_pretrained(args.clip_path)
                clip_processor = CLIPProcessor.from_pretrained(args.clip_path)

                # Sample a timestep t
                T = 1000
                t = torch.randint(0, T, (1,))  # Randomly sample a timestep

                # Generate x_t from the forward diffusion process
                x_t = forward_diffusion_sample(x_0, t)

                x_0_sensitive = estimate_x0(unet, x_t, t, image_embeds)
                x_0_neutral = estimate_x0(unet, x_t, t, neutral_image_embeds)

                inputs_sensitive = clip_processor(images=x_0_sensitive, return_tensors="pt").pixel_values
                inputs_neutral = clip_processor(images=x_0_neutral, return_tensors="pt").pixel_values

                # Get image embeddings using CLIP's image encoder
                embeddings_sensitive = clip_model.get_image_features(inputs_sensitive)
                embeddings_neutral = clip_model.get_image_features(inputs_neutral)

                # Normalize embeddings to avoid magnitude differences
                embeddings_sensitive = F.normalize(embeddings_sensitive, dim=-1)
                embeddings_neutral = F.normalize(embeddings_neutral, dim=-1)

                clip_reg_loss = criterion(embeddings_sensitive, embeddings_neutral)
                loss = loss + args.clip_reg * clip_reg_loss

            loss.backward()
            optimizer.step()

            # Add a perception loss to improve sample qulity of unet
            '''
            if args.use_perceptual:
                perceptual_unet = unet.copy()
                perceptual_unet.requires_grad_(False)
                perceptual_unet.eval()

                t = torch.randint(0, 1000)
                eps = torch.rand_like(neutral_image_embeds)
                x_t = forward(neutral_image_embeds, eps, t)

                v_pred = unet(x_t, t, text_embeddings)

                
                perceptual_loss = torch.nn.MSELoss(feature_pred, feature_real)
                perceptual_loss.backward()

                perceptual_optimizer.step()
                perceptual_optimizer.zero_grad()
            '''
            writer.add_scalar('image_training_loss', loss.item(), idx + 1)

            if (idx + 1) % args.save_iter == 0:
                save_path = unet_save_path
                save_model(unet, save_path, idx)
                print(f"save to {save_path} at iter:{idx + 1}")
            

    else:
        print("modality should be in [text, image] !")

        

if __name__ == "__main__":
    args = init_args()
    unlearning(args)

 
