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
            # Bootstrap params
            bootstrap_ratio=0.25,        # Ratio of batch to use EMA model for GT (1/8)
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
        
        # Bootstrap config
        self.bootstrap_ratio = bootstrap_ratio

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
        Compute MeanFlow loss function with bootstrap mechanism
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

        noises = torch.randn_like(images)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)
        
        bootstrap_size = int(batch_size * self.bootstrap_ratio)
        
        u_target = torch.zeros_like(v_t)
        
        u = model(z_t, r, t, **model_kwargs)
        
        # Process bootstrap samples (like, shotcut model.)
        if bootstrap_size > 0:
            ema_indices = torch.arange(bootstrap_size, device=device)
            
            ema_z_t = z_t[ema_indices]
            ema_v_t = v_t[ema_indices]
            ema_r = r[ema_indices]
            ema_t = t[ema_indices]
            ema_time_diff = time_diff[ema_indices]
            
            ema_kwargs = {}
            for k, v in model_kwargs.items():
                if torch.is_tensor(v) and v.shape[0] == batch_size:
                    ema_kwargs[k] = v[ema_indices]
                else:
                    ema_kwargs[k] = v
            
            def fn_ema(z, cur_r, cur_t):
                return ema_model(z, cur_r, cur_t, **ema_kwargs)
            
            # Compute JVP with EMA model
            primals = (ema_z_t, ema_r, ema_t)
            tangents = (ema_v_t, torch.zeros_like(ema_r), torch.ones_like(ema_t))
            _, ema_dudt = torch.func.jvp(fn_ema, primals, tangents)
            
            # Calculate bootstrap target
            ema_u_target = ema_v_t - ema_time_diff * ema_dudt
            
            # Apply CFG if needed for bootstrap samples
            ema_cfg_time_mask = (ema_t >= self.cfg_min_t) & (ema_t <= self.cfg_max_t)
            if model_kwargs.get('y') is not None and ema_cfg_time_mask.any():
                ema_y = ema_kwargs.get('y')
                ema_batch_size = ema_y.shape[0]
                num_classes = ema_model.num_classes
                
                ema_z_t_batch = torch.cat([ema_z_t, ema_z_t], dim=0)
                ema_t_batch = torch.cat([ema_t, ema_t], dim=0)
                ema_t_end_batch = torch.cat([ema_t, ema_t], dim=0)
                ema_y_batch = torch.cat([ema_y, torch.full_like(ema_y, num_classes)], dim=0)
                
                ema_combined_kwargs = ema_kwargs.copy()
                ema_combined_kwargs['y'] = ema_y_batch
                
                with torch.no_grad():
                    ema_combined_u_at_t = ema_model(ema_z_t_batch, ema_t_batch, ema_t_end_batch, **ema_combined_kwargs)
                    ema_u_cond_at_t, ema_u_uncond_at_t = torch.chunk(ema_combined_u_at_t, 2, dim=0)
                    ema_v_tilde = (self.cfg_omega * ema_v_t + 
                               self.cfg_kappa * ema_u_cond_at_t + 
                               (1 - self.cfg_omega - self.cfg_kappa) * ema_u_uncond_at_t)
                
                ema_u_target_cfg = ema_v_tilde - ema_time_diff * ema_dudt
                ema_u_target = torch.where(ema_cfg_time_mask.view(-1, 1, 1, 1), ema_u_target_cfg, ema_u_target)
            
            u_target[ema_indices] = ema_u_target
        
        # Process flow samples (use current model for GT)
        if batch_size > bootstrap_size:
            non_ema_indices = torch.arange(bootstrap_size, batch_size, device=device)
            
            non_ema_z_t = z_t[non_ema_indices]
            non_ema_v_t = v_t[non_ema_indices]
            non_ema_r = r[non_ema_indices]
            non_ema_t = t[non_ema_indices]
            non_ema_time_diff = time_diff[non_ema_indices]
            
            non_ema_kwargs = {}
            for k, v in model_kwargs.items():
                if torch.is_tensor(v) and v.shape[0] == batch_size:
                    non_ema_kwargs[k] = v[non_ema_indices]
                else:
                    non_ema_kwargs[k] = v
            
            def fn_current(z, cur_r, cur_t):
                return model(z, cur_r, cur_t, **non_ema_kwargs)
            
            # Compute JVP with current model
            primals = (non_ema_z_t, non_ema_r, non_ema_t)
            tangents = (non_ema_v_t, torch.zeros_like(non_ema_r), torch.ones_like(non_ema_t))
            _, non_ema_dudt = torch.func.jvp(fn_current, primals, tangents)
            
            non_ema_u_target = non_ema_v_t - non_ema_time_diff * non_ema_dudt
            
            # Apply CFG if needed for flow samples
            non_ema_cfg_time_mask = (non_ema_t >= self.cfg_min_t) & (non_ema_t <= self.cfg_max_t)
            if model_kwargs.get('y') is not None and non_ema_cfg_time_mask.any():
                non_ema_y = non_ema_kwargs.get('y')
                non_ema_batch_size = non_ema_y.shape[0]
                num_classes = ema_model.num_classes
                
                non_ema_z_t_batch = torch.cat([non_ema_z_t, non_ema_z_t], dim=0)
                non_ema_t_batch = torch.cat([non_ema_t, non_ema_t], dim=0)
                non_ema_t_end_batch = torch.cat([non_ema_t, non_ema_t], dim=0)
                non_ema_y_batch = torch.cat([non_ema_y, torch.full_like(non_ema_y, num_classes)], dim=0)
                
                non_ema_combined_kwargs = non_ema_kwargs.copy()
                non_ema_combined_kwargs['y'] = non_ema_y_batch
                
                with torch.no_grad():
                    non_ema_combined_u_at_t = model(non_ema_z_t_batch, non_ema_t_batch, non_ema_t_end_batch, **non_ema_combined_kwargs)
                    non_ema_u_cond_at_t, non_ema_u_uncond_at_t = torch.chunk(non_ema_combined_u_at_t, 2, dim=0)
                    non_ema_v_tilde = (self.cfg_omega * non_ema_v_t + 
                               self.cfg_kappa * non_ema_u_cond_at_t + 
                               (1 - self.cfg_omega - self.cfg_kappa) * non_ema_u_uncond_at_t)
                
                non_ema_u_target_cfg = non_ema_v_tilde - non_ema_time_diff * non_ema_dudt
                non_ema_u_target = torch.where(non_ema_cfg_time_mask.view(-1, 1, 1, 1), non_ema_u_target_cfg, non_ema_u_target)
            
            u_target[non_ema_indices] = non_ema_u_target
        
        # Detach the target to prevent gradient flow
        u_target = u_target.detach()
        
        error = u - u_target
        
        # Apply adaptive weighting based on configuration
        if self.weighting == "adaptive":
            epsilon = 1e-3
            weights = 1.0 / (torch.norm(error.reshape(error.shape[0], -1), dim=1) ** 2 + epsilon) ** self.adaptive_p
            loss = torch.mean(weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).detach() * error ** 2)
        else:
            loss = mean_flat(error ** 2)
        
        return loss