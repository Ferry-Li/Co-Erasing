python main.py \
  --modality text \
  --train_method noxattn \
  --prompt nudity \
  --devices 0,1 \
  --ckpt_path /data/feiran/stable-diffusion-v1-5


python main.py \
  --modality image \
  --train_method noxattn \
  --text_uncond \
  --prompt "nudity" \
  --devices 0,1 \
  --unet_ckpt_path "checkpoints/text/nudity/unet/unet.pth" \
  --image /data/feiran/robustDiffusion/generation_dataset_v1_5/nudity_0.85 \
  --image_number 100 \
  --text_guide "nudity" \
  --blur_factor 3 \
  --iterations 1000 \
  --negative_guidance 1.0 \
  --output_dir outputs \
  --logging_dir log \
  --save_iter 500
