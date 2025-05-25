python main.py \
  --modality text \
  --train_method noxattn \
  --prompt nudity \
  --devices 0,1 \
  --ckpt_path PATH_TO_ORIGINAL_SD


python main.py \
  --modality image \
  --train_method noxattn \
  --text_uncond \
  --prompt "nudity" \
  --devices 0,1 \
  --unet_ckpt_path PATH_TO_TEXT_ERASED_SD \
  --image PATH_TO_IMAGE_DIR \
  --image_number 100 \
  --text_guide "nudity" \
  --blur_factor 3 \
  --iterations 1000 \
  --negative_guidance 1.0 \
  --output_dir outputs \
  --logging_dir log \
  --save_iter 500
