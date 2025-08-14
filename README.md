# MeanFlow: Pytorch Implementation
This repository contains a minimalist PyTorch implementation of MeanFlow, a novel single-step flow matching model for high-quality image generation.

## DO NOT Overlook the pretrained Flow Matching model：Fine-tuning Pretrained Flow Matching Models with MeanFlow
| Model | FID(NFE=1), our results| FID(NFE=2), our results|FID(NFE=2), results in paper|
|---------------|---------------|----------------|----------------|
|SiT-XL/2(w cfg) + [pretrained weights](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) (1400 epoch)|4.52|2.81 (1400+20+40)|2.93, 240 epoch, Table 2|
|SiT-XL/2(w cfg) + [pretrained weights](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) (1400 epoch)|15.50|2.55 (1400+20+110)|2.20, 1000 epoch, Table 2|

**Tips**: Direct fine-tuning using MeanFlow with classifier-free guidance (CFG) exhibits training instability. To address this issue, we adopt a staged training strategy: initially fine-tuning with MeanFlow without CFG for 20 epochs, followed by continued fine-tuning with CFG-enabled MeanFlow. All finetuning experiments are deployed with `fp32`.

## Installation

```bash
# Clone this repository
git clone https://github.com/zhuyu-cs/MeanFlow.git
cd MeanFlow

# Install dependencies
pip install -r requirements.txt
```

## Usage

### ImageNet 256

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
    --exp-name "meanflow_xl_2" \
    --output-dir "exp" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-XL/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "fp32" \
    --epochs 80\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0\
    --cfg-omega 0.\
    --cfg-kappa 0.\
    --cfg-min-t 0.\
    --cfg-max-t -1.0\
    --finetune "/path/to/SiT-XL-2-256.pt"

```
Note: we here ues special `cfg-` parameters for finetuning pretrained sit-xl/2 without CFG.

**Sampling and Evaluation**

For large-scale sampling and quantitative evaluation (FID, IS), we provide a distributed evaluation framework:

```bash
torchrun --nproc_per_node=8 --nnodes=1 evaluate.py \
    --ckpt "/path/to/the/weights" \
    --model "SiT-XL/2" \
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
