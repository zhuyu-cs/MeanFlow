import torch
import numpy as np
import torch.func
from functools import partial

class SILoss:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            # New parameters
            time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
            time_mu=-2.0,                 # Mean parameter for logit_normal distribution
            time_sigma=2.0,               # Std parameter for logit_normal distribution
            ratio_r_not_equal_t=0.25,     # Ratio of samples where r≠t
            adaptive_p=1.0,               # Power param for adaptive weighting
            label_dropout_prob=0.1,       # Drop out label
            ):
        self.weighting = weighting
        self.path_type = path_type
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.label_dropout_prob = label_dropout_prob
        # Adaptive weight config
        self.adaptive_p = adaptive_p
        

    def interpolant(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        
        return r, t

    def __call__(self, model, images):
        """
        Compute MeanFlow loss function (unconditional)
        """
        batch_size = images.shape[0]
        device = images.device

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)
        t_ = t.view(-1, 1, 1, 1)
        r_ = r.view(-1, 1, 1, 1)
        time_diff = t_ - r_

        noises = torch.randn_like(images)
        
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_)
        z_t = alpha_t * images + sigma_t * noises
        
        v_t = d_alpha_t * images + d_sigma_t * noises
        
        def model_wrapper(z_t, r, t):
            return model.module(z_t, r, t)

        u = model(z_t, r, t) # fix bug for ddp 
        _, dudt = torch.func.jvp(
            model_wrapper,
            (z_t, r, t),  
            (v_t, torch.zeros_like(r), torch.ones_like(t)) 
        )
        
        u_target = v_t - time_diff * dudt
        
        error = u - u_target.detach()
        
        if self.weighting == "adaptive":
            error_norm = torch.norm(error.reshape(error.shape[0], -1), dim=1)
            weights = 1.0 / (error_norm.detach() ** 2 + 1e-3).pow(self.adaptive_p)
            loss = weights * error_norm ** 2
        else:
            error_norm = torch.norm(error.reshape(error.shape[0], -1), dim=1)
            loss = error_norm ** 2
        
        return loss
