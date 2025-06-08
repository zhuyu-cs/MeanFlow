import torch
import numpy as np
import torch.func

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

        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y'].clone()  
            batch_size = y.shape[0]
            num_classes = ema_model.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes
            model_kwargs['y'] = y
            unconditional_mask = dropout_mask  # Used for unconditional velocity computation

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)

        noises = torch.randn_like(images)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)
        
        bootstrap_size = int(batch_size * self.bootstrap_ratio)
        
        u_target = torch.zeros_like(v_t)
        
        u = model(z_t, r, t, **model_kwargs)
        
        # Process bootstrap samples (using EMA model)
        if bootstrap_size > 0:
            ema_indices = torch.arange(bootstrap_size, device=device)
            
            ema_z_t = z_t[ema_indices]
            ema_v_t = v_t[ema_indices]
            ema_r = r[ema_indices]
            ema_t = t[ema_indices]
            ema_time_diff = time_diff[ema_indices]
            ema_unconditional_mask = unconditional_mask[ema_indices]
            
            ema_kwargs = {}
            for k, v in model_kwargs.items():
                if torch.is_tensor(v) and v.shape[0] == batch_size:
                    ema_kwargs[k] = v[ema_indices]
                else:
                    ema_kwargs[k] = v
            
            def fn_ema(z, cur_r, cur_t):
                return ema_model(z, cur_r, cur_t, **ema_kwargs)
            
            # Check if CFG should be applied (exclude unconditional samples)
            ema_cfg_time_mask = (ema_t >= self.cfg_min_t) & (ema_t <= self.cfg_max_t) & (~ema_unconditional_mask)
            
            if model_kwargs.get('y') is not None and ema_cfg_time_mask.any():
                # Split samples into CFG and non-CFG
                ema_cfg_indices = torch.where(ema_cfg_time_mask)[0]
                ema_no_cfg_indices = torch.where(~ema_cfg_time_mask)[0]
                
                ema_u_target = torch.zeros_like(ema_v_t)
                
                # Process CFG samples
                if len(ema_cfg_indices) > 0:
                    ema_cfg_z_t = ema_z_t[ema_cfg_indices]
                    ema_cfg_v_t = ema_v_t[ema_cfg_indices]
                    ema_cfg_r = ema_r[ema_cfg_indices]
                    ema_cfg_t = ema_t[ema_cfg_indices]
                    ema_cfg_time_diff = ema_time_diff[ema_cfg_indices]
                    
                    ema_cfg_kwargs = {}
                    for k, v in ema_kwargs.items():
                        if torch.is_tensor(v) and v.shape[0] == len(ema_indices):
                            ema_cfg_kwargs[k] = v[ema_cfg_indices]
                        else:
                            ema_cfg_kwargs[k] = v
                    
                    # Compute v_tilde for CFG samples
                    ema_cfg_y = ema_cfg_kwargs.get('y')
                    num_classes = ema_model.num_classes
                    
                    ema_cfg_z_t_batch = torch.cat([ema_cfg_z_t, ema_cfg_z_t], dim=0)
                    ema_cfg_t_batch = torch.cat([ema_cfg_t, ema_cfg_t], dim=0)
                    ema_cfg_t_end_batch = torch.cat([ema_cfg_t, ema_cfg_t], dim=0)
                    ema_cfg_y_batch = torch.cat([ema_cfg_y, torch.full_like(ema_cfg_y, num_classes)], dim=0)
                    
                    ema_cfg_combined_kwargs = ema_cfg_kwargs.copy()
                    ema_cfg_combined_kwargs['y'] = ema_cfg_y_batch
                    
                    with torch.no_grad():
                        ema_cfg_combined_u_at_t = ema_model(ema_cfg_z_t_batch, ema_cfg_t_batch, ema_cfg_t_end_batch, **ema_cfg_combined_kwargs)
                        ema_cfg_u_cond_at_t, ema_cfg_u_uncond_at_t = torch.chunk(ema_cfg_combined_u_at_t, 2, dim=0)
                        ema_cfg_v_tilde = (self.cfg_omega * ema_cfg_v_t + 
                                self.cfg_kappa * ema_cfg_u_cond_at_t + 
                                (1 - self.cfg_omega - self.cfg_kappa) * ema_cfg_u_uncond_at_t)
                    
                    # Compute JVP with CFG velocity
                    def fn_ema_cfg(z, cur_r, cur_t):
                        return ema_model(z, cur_r, cur_t, **ema_cfg_kwargs)
                    
                    primals = (ema_cfg_z_t, ema_cfg_r, ema_cfg_t)
                    tangents = (ema_cfg_v_tilde, torch.zeros_like(ema_cfg_r), torch.ones_like(ema_cfg_t))
                    _, ema_cfg_dudt = torch.func.jvp(fn_ema_cfg, primals, tangents)
                    
                    ema_cfg_u_target = ema_cfg_v_tilde - ema_cfg_time_diff * ema_cfg_dudt
                    ema_u_target[ema_cfg_indices] = ema_cfg_u_target
                
                # Process non-CFG samples (including unconditional ones)
                if len(ema_no_cfg_indices) > 0:
                    ema_no_cfg_z_t = ema_z_t[ema_no_cfg_indices]
                    ema_no_cfg_v_t = ema_v_t[ema_no_cfg_indices]
                    ema_no_cfg_r = ema_r[ema_no_cfg_indices]
                    ema_no_cfg_t = ema_t[ema_no_cfg_indices]
                    ema_no_cfg_time_diff = ema_time_diff[ema_no_cfg_indices]
                    
                    ema_no_cfg_kwargs = {}
                    for k, v in ema_kwargs.items():
                        if torch.is_tensor(v) and v.shape[0] == len(ema_indices):
                            ema_no_cfg_kwargs[k] = v[ema_no_cfg_indices]
                        else:
                            ema_no_cfg_kwargs[k] = v
                    
                    def fn_ema_no_cfg(z, cur_r, cur_t):
                        return ema_model(z, cur_r, cur_t, **ema_no_cfg_kwargs)
                    
                    primals = (ema_no_cfg_z_t, ema_no_cfg_r, ema_no_cfg_t)
                    tangents = (ema_no_cfg_v_t, torch.zeros_like(ema_no_cfg_r), torch.ones_like(ema_no_cfg_t))
                    _, ema_no_cfg_dudt = torch.func.jvp(fn_ema_no_cfg, primals, tangents)
                    
                    ema_no_cfg_u_target = ema_no_cfg_v_t - ema_no_cfg_time_diff * ema_no_cfg_dudt
                    ema_u_target[ema_no_cfg_indices] = ema_no_cfg_u_target
            else:
                # No labels or no CFG applicable samples, use standard JVP
                primals = (ema_z_t, ema_r, ema_t)
                tangents = (ema_v_t, torch.zeros_like(ema_r), torch.ones_like(ema_t))
                _, ema_dudt = torch.func.jvp(fn_ema, primals, tangents)
                
                ema_u_target = ema_v_t - ema_time_diff * ema_dudt
            
            u_target[ema_indices] = ema_u_target
        
        # Process non-bootstrap samples (using current model)
        if batch_size > bootstrap_size:
            non_ema_indices = torch.arange(bootstrap_size, batch_size, device=device)
            
            non_ema_z_t = z_t[non_ema_indices]
            non_ema_v_t = v_t[non_ema_indices]
            non_ema_r = r[non_ema_indices]
            non_ema_t = t[non_ema_indices]
            non_ema_time_diff = time_diff[non_ema_indices]
            non_ema_unconditional_mask = unconditional_mask[non_ema_indices]
            
            non_ema_kwargs = {}
            for k, v in model_kwargs.items():
                if torch.is_tensor(v) and v.shape[0] == batch_size:
                    non_ema_kwargs[k] = v[non_ema_indices]
                else:
                    non_ema_kwargs[k] = v
            
            def fn_current(z, cur_r, cur_t):
                return model(z, cur_r, cur_t, **non_ema_kwargs)
            
            # Check if CFG should be applied (exclude unconditional samples)
            non_ema_cfg_time_mask = (non_ema_t >= self.cfg_min_t) & (non_ema_t <= self.cfg_max_t) & (~non_ema_unconditional_mask)
            
            if model_kwargs.get('y') is not None and non_ema_cfg_time_mask.any():
                # Split samples into CFG and non-CFG
                non_ema_cfg_indices = torch.where(non_ema_cfg_time_mask)[0]
                non_ema_no_cfg_indices = torch.where(~non_ema_cfg_time_mask)[0]
                
                non_ema_u_target = torch.zeros_like(non_ema_v_t)
                
                # Process CFG samples
                if len(non_ema_cfg_indices) > 0:
                    non_ema_cfg_z_t = non_ema_z_t[non_ema_cfg_indices]
                    non_ema_cfg_v_t = non_ema_v_t[non_ema_cfg_indices]
                    non_ema_cfg_r = non_ema_r[non_ema_cfg_indices]
                    non_ema_cfg_t = non_ema_t[non_ema_cfg_indices]
                    non_ema_cfg_time_diff = non_ema_time_diff[non_ema_cfg_indices]
                    
                    non_ema_cfg_kwargs = {}
                    for k, v in non_ema_kwargs.items():
                        if torch.is_tensor(v) and v.shape[0] == len(non_ema_indices):
                            non_ema_cfg_kwargs[k] = v[non_ema_cfg_indices]
                        else:
                            non_ema_cfg_kwargs[k] = v
                    
                    # Compute v_tilde for CFG samples
                    non_ema_cfg_y = non_ema_cfg_kwargs.get('y')
                    num_classes = ema_model.num_classes
                    
                    non_ema_cfg_z_t_batch = torch.cat([non_ema_cfg_z_t, non_ema_cfg_z_t], dim=0)
                    non_ema_cfg_t_batch = torch.cat([non_ema_cfg_t, non_ema_cfg_t], dim=0)
                    non_ema_cfg_t_end_batch = torch.cat([non_ema_cfg_t, non_ema_cfg_t], dim=0)
                    non_ema_cfg_y_batch = torch.cat([non_ema_cfg_y, torch.full_like(non_ema_cfg_y, num_classes)], dim=0)
                    
                    non_ema_cfg_combined_kwargs = non_ema_cfg_kwargs.copy()
                    non_ema_cfg_combined_kwargs['y'] = non_ema_cfg_y_batch
                    
                    with torch.no_grad():
                        non_ema_cfg_combined_u_at_t = model(non_ema_cfg_z_t_batch, non_ema_cfg_t_batch, non_ema_cfg_t_end_batch, **non_ema_cfg_combined_kwargs)
                        non_ema_cfg_u_cond_at_t, non_ema_cfg_u_uncond_at_t = torch.chunk(non_ema_cfg_combined_u_at_t, 2, dim=0)
                        non_ema_cfg_v_tilde = (self.cfg_omega * non_ema_cfg_v_t + 
                                self.cfg_kappa * non_ema_cfg_u_cond_at_t + 
                                (1 - self.cfg_omega - self.cfg_kappa) * non_ema_cfg_u_uncond_at_t)
                    
                    # Compute JVP with CFG velocity
                    def fn_current_cfg(z, cur_r, cur_t):
                        return model(z, cur_r, cur_t, **non_ema_cfg_kwargs)
                    
                    primals = (non_ema_cfg_z_t, non_ema_cfg_r, non_ema_cfg_t)
                    tangents = (non_ema_cfg_v_tilde, torch.zeros_like(non_ema_cfg_r), torch.ones_like(non_ema_cfg_t))
                    _, non_ema_cfg_dudt = torch.func.jvp(fn_current_cfg, primals, tangents)
                    
                    non_ema_cfg_u_target = non_ema_cfg_v_tilde - non_ema_cfg_time_diff * non_ema_cfg_dudt
                    non_ema_u_target[non_ema_cfg_indices] = non_ema_cfg_u_target
                
                # Process non-CFG samples (including unconditional ones)
                if len(non_ema_no_cfg_indices) > 0:
                    non_ema_no_cfg_z_t = non_ema_z_t[non_ema_no_cfg_indices]
                    non_ema_no_cfg_v_t = non_ema_v_t[non_ema_no_cfg_indices]
                    non_ema_no_cfg_r = non_ema_r[non_ema_no_cfg_indices]
                    non_ema_no_cfg_t = non_ema_t[non_ema_no_cfg_indices]
                    non_ema_no_cfg_time_diff = non_ema_time_diff[non_ema_no_cfg_indices]
                    
                    non_ema_no_cfg_kwargs = {}
                    for k, v in non_ema_kwargs.items():
                        if torch.is_tensor(v) and v.shape[0] == len(non_ema_indices):
                            non_ema_no_cfg_kwargs[k] = v[non_ema_no_cfg_indices]
                        else:
                            non_ema_no_cfg_kwargs[k] = v
                    
                    def fn_current_no_cfg(z, cur_r, cur_t):
                        return model(z, cur_r, cur_t, **non_ema_no_cfg_kwargs)
                    
                    primals = (non_ema_no_cfg_z_t, non_ema_no_cfg_r, non_ema_no_cfg_t)
                    tangents = (non_ema_no_cfg_v_t, torch.zeros_like(non_ema_no_cfg_r), torch.ones_like(non_ema_no_cfg_t))
                    _, non_ema_no_cfg_dudt = torch.func.jvp(fn_current_no_cfg, primals, tangents)
                    
                    non_ema_no_cfg_u_target = non_ema_no_cfg_v_t - non_ema_no_cfg_time_diff * non_ema_no_cfg_dudt
                    non_ema_u_target[non_ema_no_cfg_indices] = non_ema_no_cfg_u_target
            else:
                # No labels or no CFG applicable samples, use standard JVP
                primals = (non_ema_z_t, non_ema_r, non_ema_t)
                tangents = (non_ema_v_t, torch.zeros_like(non_ema_r), torch.ones_like(non_ema_t))
                _, non_ema_dudt = torch.func.jvp(fn_current, primals, tangents)
                
                non_ema_u_target = non_ema_v_t - non_ema_time_diff * non_ema_dudt
            
            u_target[non_ema_indices] = non_ema_u_target
        
        # Detach the target to prevent gradient flow
        u_target = u_target.detach()
        
        error = u - u_target
        
        # Apply adaptive weighting based on configuration
        if self.weighting == "adaptive":
            error_norm = torch.norm(error.reshape(error.shape[0], -1), dim=1)
            weights = 1.0 / (error_norm.detach() ** 2 + 1e-3).pow(self.adaptive_p)
            loss = weights * error_norm ** 2
        else:
            error_norm = torch.norm(error.reshape(error.shape[0], -1), dim=1)
            loss = error_norm ** 2
        
        return loss