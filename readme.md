# Co-Erasing: Collaborative Erasing with Text-Image Prompts

This is the official code for the paper "One Image is Worth a Thousand Words: A Usability-Preservable Text-Image Collaborative Erasing Framework" accepted by International Conference on Machine Learning (ICML2025).

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2505.11131)    [![checkpoints](https://img.shields.io/badge/hugging%20face-checkpoint-1082c3)](https://huggingface.co/Ferry30/Co-Erasing)  [![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://github.com/Ferry-Li/Co-Erasing)

**Paper Title: One Image is Worth a Thousand Words: A Usability-Preservable Text-Image Collaborative Erasing Framework**

**Authors:** [Feiran Li](https://ferry-li.github.io/), [Qianqian Xu\*](https://qianqianxu010.github.io/), [Shilong Bao](https://statusrank.github.io/), [Zhiyong Yang](https://joshuaas.github.io/), [Xiaochun Cao](https://scst.sysu.edu.cn/members/1401493.htm), [Qingming Huang\*](https://people.ucas.ac.cn/~qmhuang)

![example](example.png)

## Installation

- Some codes and checkpoints are under examination, and we will update as soon as possible.
- We update some checkpoints [here](https://huggingface.co/Ferry30/Co-Erasing). The models are based on [ESD](https://github.com/rohitgandikota/erasing).

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
  python main.py \
  --modality text \
  --train_method noxattn \
  --prompt nudity \
  --devices 0,1 \
  --ckpt_path PATH_TO_SD
  ```

- Step3: Conduct Co-erasing:

  ```bash
  python main.py \
  --modality image \
  --train_method full \
  --text_uncond \
  --prompt "nudity" \
  --devices 2,3 \
  --unet_ckpt_path PATH_TO_TEXT_ERASED_SD \
  --image PATH_TO_IMAGE_DIR \
  --image_number 200 \
  --text_guide "nudity" \
  --blur_factor 5 \
  --iterations 1500 \
  --negative_guidance 1.0 \
  --output_dir outputs \
  --logging_dir log \
  --save_iter 500
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
