python erase_pipeline_v1_5.py --modality text --prompt 'bird' --train_method noxattn --devices 0,1 --iterations 1000

python erase_pipeline_v1_5.py --modality image --train_method noxattn --text_uncond --prompt 'bird' --devices 0,1 --unet_ckpt_path "checkpoints/text/bird/unet/unet.pth" --image generation_dataset_v1_5/bird --image_number 100 --text_guide bird --blur_factor 3 --iterations 3000 --negative_guidance 1.0
