import torch


def get_training_params(unet, train_method):
    params = []
    for name, param in unet.named_parameters():
        if train_method == 'full':
            params.append(param)
        elif train_method == 'noxattn' and not ('attn2' in name or name.startswith('out.') or 'time_embed' in name):
            params.append(param)
        elif train_method == 'selfattn' and 'attn1' in name:
            params.append(param)
        elif train_method == 'xattn' and 'attn2' in name:
            params.append(param)
        elif train_method == 'notime' and not name.startswith('time_embed'):
            params.append(param)
        elif train_method == 'xlayer' and 'attn2' in name and ('output_blocks.6.' in name or 'output_blocks.8.' in name):
            params.append(param)
        elif train_method == 'selflayer' and 'attn1' in name and ('input_blocks.4.' in name or 'input_blocks.7.' in name):
            params.append(param)
    return params


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


def anneal_schedule(timestep, total_steps):
    x = torch.tensor(timestep / total_steps)
    return 1 / torch.exp(x)
