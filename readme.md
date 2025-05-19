# Co-Erasing: Collaborative Erasing with Text-Image Prompts

This is the official code for the paper "One Image is Worth a Thousand Words: A Usability-Preservable Text-Image Collaborative Erasing Framework" accepted by International Conference on Machine Learning (ICML2025). [paper](https://arxiv.org/abs/2505.11131).

**Paper Title: One Image is Worth a Thousand Words: A Usability-Preservable Text-Image Collaborative Erasing Framework**

**Authors:** [Feiran Li](https://ferry-li.github.io/), [Qianqian Xu\*](https://qianqianxu010.github.io/), [Shilong Bao](https://statusrank.github.io/), [Zhiyong Yang](https://joshuaas.github.io/), [Xiaochun Cao](https://scst.sysu.edu.cn/members/1401493.htm), [Qingming Huang\*](https://people.ucas.ac.cn/~qmhuang)

![example](example.png)

## Installation

- Some codes and arxiv paper is under construction and examination, and we will update as soon as possible.

Clone this repository:

```bash
git clone git@github.com:Ferry-Li/Co-Erasing.git
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

# Training

Before training, there are a few steps.

- Step1: Generate reference images

  ```bash
  python generate_data.py --label nudity
  ```

- Step2: Acquire a pretrained text-based erased model (such as ESD), or simply run

  ```bash
  python erase_pipeline.py --modality text --prompt 'nudity' --train_method noxattn  --iterations 1000
  ```

- Step3: Conduct Co-erasing:

  ```bash
  python erase_pipeline.py --modality image --train_method noxattn --text_uncond --prompt 'nudity' --devices 0,1 --unet_ckpt_path "checkpoints/text/bird/unet/unet.pth" --image generation_dataset/nudity --image_number 100 --text_guide nudity --blur_factor 3 --iterations 1000 --negative_guidance 1.0
  ```

# Evaluation

The evaluation follows [UnlearnDiff](https://github.com/OPTML-Group/Diffusion-MU-Attack).

## Citation

If you find this work or repository useful, please cite the following:

```bib
@inproceedings{li2024coerasing,
title={One Image is Worth a Thousand Words: A Usability-Preservable Text-Image Collaborative Erasing Framework}, 
author={Feiran Li and Qianqian Xu and Shilong Bao and Zhiyong Yang and Xiaochun Cao and Qingming Huang},
booktitle={The Forty-first International Conference on Machine Learning},
year={2025}
}
```

## Contact us

If you have any detailed questions or suggestions, feel free to email us: [lifeiran@iie.ac.cn](mailto:lifeiran@iie.ac.cn)! Thanks for your interest in our work!
