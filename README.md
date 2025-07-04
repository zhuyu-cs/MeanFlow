# MeanFlow: Pytorch Implementation
This repository contains a minimalist PyTorch implementation of MeanFlow, a novel single-step flow matching model for high-quality image generation.

## Overview
MeanFlow introduces a principled framework for one-step generative modeling by formulating the concept of average velocity in Flow Matching methods. In contrast to conventional approaches that model instantaneous velocities, MeanFlow leverages the MeanFlow Identity to establish a well-defined relationship between average and instantaneous velocities, enabling robust single-step generation without requiring pre-training, distillation, or curriculum learning.

Built upon the [SiT](https://github.com/willisma/SiT/tree/main)  transformer architecture, this implementation focuses on reproducing the state-of-the-art single-step generation capabilities demonstrated in the original paper. 

## Reproduced ImageNet Results

| Model | Epoch | FID(NFE=1), our results| FID(NFE=1), results in paper|
|---------------|---------------|----------------|----------------|
|SiT-B/4(no cfg)| 80 |58.74|61.06, Table 1f|
|SiT-B/4(w cfg)| 80 |15.43|15.53, Table 1f|
|SiT-B/2(w cfg)| 240 |6.06|6.17, Table 2|
|SiT-L/2(w cfg)| 240 |4.12(130/240) *training*|3.84, Table 2|
|SiT-XL/2(w cfg)| 240 |3.56(140/240) *training*|3.43, Table 2|
|SiT-XL/2(w cfg)| 240 |22.97(20/240) *training*|ImageNet512|

**Note**: **We are still working on reproducing all experimental results and plan to release the trained model weights upon completion**.
For comprehensive performance metrics and theories, please refer to the original paper: [Mean Flows for One-step Generative Modeling](https://arxiv.org/pdf/2505.13447).

Other exploration：Fine-tuning Pretrained Flow Matching Models
| Model | FID(NFE=1), our results| FID(NFE=2), our results|FID(NFE=2), results in paper|
|---------------|---------------|----------------|----------------|
|SiT-XL/2(w cfg) + [pretrained weights](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) (1400 epoch)|4.52|2.81 (1400+20+40)|2.93, 240 epoch, Table 2|
|SiT-XL/2(w cfg) + [pretrained weights](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) (1400 epoch)|15.50|2.55 (1400+20+110)|2.20, 1000 epoch, Table 2|

**Tips**: Direct fine-tuning using MeanFlow with classifier-free guidance (CFG) exhibits training instability. To address this issue, we adopt a staged training strategy: initially fine-tuning with MeanFlow without CFG for 20 epochs, followed by continued fine-tuning with CFG-enabled MeanFlow.


**Notes**: 
1. When evaluating models trained with CFG , the --cfg-scale parameter must be set to 1.0 during inference, as the CFG guidance has been incorporated into the model during training and is no longer controllable at sampling time.
2. We currently use [sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-mse), which is not the suggested tokenizer in original paper ([sd-vae-ft-mse](https://huggingface.co/pcuenq/sd-vae-ft-mse-flax)). **Maybe replacing with ```sd-vae-ft-mse``` would yield better results**.

## Installation

```bash
# Clone this repository
git clone https://github.com/zhuyu-cs/MeanFlow.git
cd MeanFlow

# Install dependencies
pip install -r requirements.txt
```

## Usage

### CIFAR10

**Requirements**
- NVIDIA A100/H100 80GB GPU recommended for optimal performance

*Note: The UNet architecture needs higher memory consumption compared to Diffusion Transformer (DiT) models*

**Training**

1. Switch to the CIFAR-10 experimental branch:
```bash
git checkout cifar10
```

2. Standard Training (High Memory)
```bash
accelerate launch --num_processes=8 \
    train.py \
    --exp-name "cifar_unet" \
    --output-dir "work_dir" \
    --data-dir "/data/dataset/train_sdvae_latents_lmdb" \
    --resolution 32 \
    --batch-size 1024 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 19200\ # about 800k iters.
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -2.0 \
    --time-sigma 2.0 \
    --ratio-r-not-equal-t 0.75 \
    --adaptive-p 0.75
```

2. Memory-Efficient Training (Lower GPU Memory)
```bash
accelerate launch --num_processes=8 \
      train.py \
      --exp-name "cifar_unet" \
      --output-dir "work_dir" \
      --data-dir "/data/dataset/train_sdvae_latents_lmdb" \
      --resolution 32 \
      --batch-size 512 \
      --gradient-accumulation-steps 2 \
      --allow-tf32 \
      --mixed-precision "bf16" \
      --epochs 19200\ 
      --path-type "linear" \
      --weighting "adaptive" \
      --time-sampler "logit_normal" \
      --time-mu -2.0 \
      --time-sigma 2.0 \
      --ratio-r-not-equal-t 0.75 \
      --adaptive-p 0.75
```

3. Evaluation 
```bash
torchrun --nproc_per_node=8 evaluate.py \
    --ckpt "./work_dir/cifar_unet/checkpoints/0200000.pt" \
    --per-proc-batch-size 128 \
    --num-fid-samples 50000 \
    --sample-dir "./fid_dir" \
    --compute-metrics \
    --num-steps 1\
    --fid-ref "train"
```
**Results**

| Iters | FID(NFE=1)|
|---------------|----------------|
| 50k|210.36|
| 100k|6.35|



### ImageNet 256

**Switch to ImageNet branch**
```bash
git checkout main
```

**Preparing Data**

This implementation utilizes LMDB datasets with VAE-encoded latent representations for efficient training. The preprocessing pipeline is adapted from the [MAR](https://github.com/LTH14/mar/blob/main/main_cache.py).

```bash
# Example dataset preparation for ImageNet
cd ./preprocess_imagenet
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    main_cache.py \
    --source_lmdb /data/ImageNet_train \
    --target_lmdb /data/train_vae_latents_lmdb \
    --img_size 256 \
    --batch_size 1024 \
    --lmdb_size_gb 400
```
*Note: The preprocessing assumes ImageNet has been pre-converted to LMDB format.*

**Training**

We provide training configurations for different model scales (B, L, XL) based on the hyperparameters from the original paper::

```bash

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_b_4" \
    --output-dir "work_dir" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-B/4" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 80\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 3.0 \ #1.0 for no cfg
    --cfg-kappa 0.\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_b_2" \
    --output-dir "exp" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 1.0 \
    --cfg-kappa 0.5\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_l_2" \
    --output-dir "exp" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-L/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 0.2 \
    --cfg-kappa 0.92\
    --cfg-min-t 0.0\
    --cfg-max-t 0.8

```
Each configuration is optimized for different model sizes according to the original paper's settings.

**Sampling and Evaluation**

For large-scale sampling and quantitative evaluation (FID, IS), we provide a distributed evaluation framework:

```bash
torchrun --nproc_per_node=8 --nnodes=1 evaluate.py \
    --ckpt "/path/to/the/weights" \
    --model "SiT-L/2" \
    --resolution 256 \
    --cfg-scale 1.0 \
    --per-proc-batch-size 128 \
    --num-fid-samples 50000 \
    --sample-dir "./fid_dir" \
    --compute-metrics \
    --num-steps 1\
    --fid-statistics-file "./fid_stats/adm_in256_stats.npz"
```
This evaluation performs distributed sampling across 8 GPUs to generate 50,000 high-quality samples for robust FID computation. The framework validates MeanFlow's single-step generation capability (num-steps=1) and computes FID scores against pre-computed ImageNet statistics.

## Acknowledgements

This implementation builds upon:
- [SiT](https://github.com/willisma/SiT/tree/main) (model architecture)
- [REPA](https://github.com/sihyun-yu/REPA/tree/main) (training pipeline)
- [MAR](https://github.com/LTH14/mar/tree/main) (data preprocessing)

## Citation
If you find this implementation useful in your research, please cite the original work and this repo:
```
@article{geng2025mean,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhengyang and Deng, Mingyang and Bai, Xingjian and Kolter, J Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}

@misc{meanflow_pytorch,
  title={MeanFlow: PyTorch Implementation},
  author={Zhu, Yu},
  year={2025},
  howpublished={\url{https://github.com/zhuyu-cs/MeanFlow}},
  note={PyTorch implementation of Mean Flows for One-step Generative Modeling}
}
```
## License

[MIT License](LICENSE)
