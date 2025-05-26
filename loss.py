import torch
import numpy as np
import torch.func

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            # New parameters
            time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
            time_mu=-0.4,                 # Mean parameter for logit_normal distribution
            time_sigma=1.0,               # Std parameter for logit_normal distribution
            ratio_r_not_equal_t=0.75,     # Ratio of samples where r≠t
            adaptive_p=1.0,               # Power param for adaptive weighting
            label_dropout_prob=0.1,       # Drop out label
            # CFG related params
            cfg_omega=1.0,                # CFG omega param, default 1.0 means no CFG
            cfg_kappa=0.0,                # CFG kappa param for mixing class-cond and uncond u
            cfg_min_t=0.0,                # Minium CFG trigger time 
            cfg_max_t=0.8,                # Maximum CFG trigger time
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
        
        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t

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

    def __call__(self, model, ema_model, images, model_kwargs=None):
        """
        Compute MeanFlow loss function
        """
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        batch_size = images.shape[0]
        device = images.device

        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y'].clone()  
            batch_size = y.shape[0]
            num_classes = ema_model.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes
        
            model_kwargs['y'] = y

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)

        # Generate noise
        noises = torch.randn_like(images)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises
        
        # Define function for JVP computation
        def fn(z, cur_r, cur_t):
            return ema_model(z, cur_r, cur_t, **model_kwargs)
        
        # Compute u and du/dt using JVP
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        _, dudt = torch.func.jvp(fn, primals, tangents) # ema model for gt
        u = model(z_t, r, t, **model_kwargs)
        # Calculate MeanFlow target
        time_diff = (t - r).view(-1, 1, 1, 1)
        
        # default: wo cfg.
        u_target = v_t - time_diff * dudt
        
        # use cfg with t condition.
        cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t)
        if model_kwargs.get('y') is not None and cfg_time_mask.any():
            y = model_kwargs['y']
            batch_size = y.shape[0]
            num_classes = ema_model.num_classes

            z_t_batch = torch.cat([z_t, z_t], dim=0)
            t_batch = torch.cat([t, t], dim=0)
            t_end_batch = torch.cat([t, t], dim=0)
            y_batch = torch.cat([y, torch.full_like(y, num_classes)], dim=0)
            combined_kwargs = model_kwargs.copy()
            combined_kwargs['y'] = y_batch
            with torch.no_grad():
                combined_u_at_t = ema_model(z_t_batch, t_batch, t_end_batch, **combined_kwargs)
                # u_θ^cfg(z_t, t, t|c), class-conditional; u_θ^cfg(z_t, t, t), class-unconditional
                u_cond_at_t, u_uncond_at_t = torch.chunk(combined_u_at_t, 2, dim=0)
                # Eq-21：ṽ_t = ω * v_t + κ * u_θ^cfg(z_t, t, t | c) + (1-ω-κ) * u_θ^cfg(z_t, t, t)
                v_tilde = (self.cfg_omega * v_t + 
                        self.cfg_kappa * u_cond_at_t + 
                        (1 - self.cfg_omega - self.cfg_kappa) * u_uncond_at_t)
            u_target_cfg = v_tilde - time_diff * dudt
            
            u_target = torch.where(cfg_time_mask.view(-1, 1, 1, 1), u_target_cfg, u_target)
        
        # Stop gradient propagation
        u_target = u_target.detach()
        
        # Calculate error
        error = u - u_target
        
        # Apply adaptive weighting based on configuration
        if self.weighting == "adaptive":
            epsilon = 1e-3
            weights = 1.0 / (torch.norm(error.reshape(error.shape[0], -1), dim=1) ** 2 + epsilon) ** self.adaptive_p
            loss = torch.mean(weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).detach() * error ** 2)
        else:
            loss = mean_flat(error ** 2)
        
        return loss