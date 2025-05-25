import torch

def denoise_to_text_timestep(unet, text_embeddings, t, start_code, noise_scheduler):
    latents = start_code
    timesteps = noise_scheduler.timesteps
    for i, step in enumerate(reversed(timesteps)):
        t_tensor = torch.tensor([step], dtype=torch.long, device=unet.device)
        with torch.no_grad():
            noise_pred = unet(latents, t_tensor, encoder_hidden_states=text_embeddings).sample
        latents = noise_scheduler.step(noise_pred, t_tensor, latents).prev_sample
        if i == t:
            break
    return latents

def predict_text_t_noise(z, t_ddpm, unet, text_embeddings):
    noise_pred = unet(
                z.to(unet.device),
                t_ddpm.to(unet.device),
                encoder_hidden_states=text_embeddings.to(unet.device),
                return_dict=False,
            )[0]
    return noise_pred

def predict_image_t_noise(z, t_ddpm, unet, text_embedding, ip_adapter, image_embeds, schedule=None):
    noise_pred = ip_adapter(z.to(unet.device), t_ddpm.to(unet.device), text_embedding.to(unet.device), image_embeds.to(unet.device), schedule)

    return noise_pred



def set_scheduler_device(scheduler, device):
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.one = scheduler.one.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
