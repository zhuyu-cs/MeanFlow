import os
import argparse
import json
import numpy as np
import math
from tqdm import tqdm
from PIL import Image

import torch
import torch.distributed as dist
import torch_fidelity

from unet import SongUNet
from meanflow_sampler import meanflow_sampler


def main(args):
    """
    Run sampling and evaluation for unconditional CIFAR-10.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    model = SongUNet(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=0,  # Unconditional
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=f'cuda:{device}')
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create folder to save samples:
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    folder_name = f"meanflow-cifar10-unconditional-{ckpt_string_name}-" \
                  f"steps-{args.num_steps}-seed-{args.global_seed}"
    eval_fid_dir = f"{args.sample_dir}/{folder_name}"
    img_folder = os.path.join(eval_fid_dir, 'images')
    if rank == 0:
        os.makedirs(eval_fid_dir, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        print(f"Saving .png samples at {eval_fid_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Using {args.num_steps}-step sampling")
    
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
    for _ in pbar:
        # Sample noise at full resolution for CIFAR-10
        z = torch.randn(n, 3, 32, 32, device=device)

        # Sample images using MeanFlow (unconditional):
        with torch.no_grad():
            samples = meanflow_sampler(
                model=model, 
                latents=z,
                cfg_scale=1.0,  # No CFG for unconditional
                num_steps=args.num_steps,
            )
            
            # Convert to [0, 255] range
            samples = (samples + 1) / 2.0
            samples = torch.clamp(255.0 * samples, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{img_folder}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    
    # Calculate FID and IS metrics (only on rank 0)
    if rank == 0 and args.compute_metrics:
        print(f"Computing evaluation metrics...")
        
        metrics_args = {
            'input1': img_folder,
            'input2': 'cifar10-train' if args.fid_ref == 'train' else 'cifar10-test',
            'cuda': True,
            'isc': True,
            'fid': True,
            'kid': False,
            'verbose': True,
        }
        
        metrics_dict = torch_fidelity.calculate_metrics(**metrics_args)
        
        fid = metrics_dict.get('frechet_inception_distance', None)
        is_mean = metrics_dict.get('inception_score_mean', None)
        is_std = metrics_dict.get('inception_score_std', None)
        
        print(f"\n===== Evaluation Results =====")
        if fid is not None:
            print(f"FID: {fid:.2f}")
        if is_mean is not None:
            print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
            
        # Save results
        results = {
            'fid': fid,
            'inception_score_mean': is_mean,
            'inception_score_std': is_std,
            'num_samples': total_samples,
            'num_steps': args.num_steps,
            'checkpoint': args.ckpt,
        }
        
        metrics_file = os.path.join(eval_fid_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Metrics saved to {metrics_file}")
        
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)
    # logging/saving:
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a MeanFlow checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # sampling
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--num-steps", type=int, default=1, help="Number of sampling steps")
    
    # Evaluation metrics
    parser.add_argument("--compute-metrics", action="store_true", help="Compute FID and IS after sampling")
    parser.add_argument("--fid-ref", type=str, default="train", choices=["train", "test"],
                       help="Reference dataset for FID computation")

    args = parser.parse_args()
    
    main(args)