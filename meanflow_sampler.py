import torch

@torch.no_grad()
def meanflow_sampler(
    model, 
    latents, 
    y=None, 
    cfg_scale=1.0,
    num_steps=1, 
    **kwargs
):
    """
    MeanFlow sampler supporting both single-step and multi-step generation
    
    Based on Eq.(12): z_r = z_t - (t-r)u(z_t, r, t)
    For single-step: z_0 = z_1 - u(z_1, 0, 1)
    For multi-step: iteratively apply the Eq.(12) with intermediate steps
    """
    batch_size = latents.shape[0]
    device = latents.device
    
    # Prepare for CFG if needed
    do_cfg = y is not None and cfg_scale > 1.0
    if do_cfg:
        if hasattr(model, 'module'):  # DDP
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes
        null_y = torch.full_like(y, num_classes)
    
    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        
        if do_cfg:
            z_combined = torch.cat([latents, latents], dim=0)
            r_combined = torch.cat([r, r], dim=0)
            t_combined = torch.cat([t, t], dim=0)
            y_combined = torch.cat([y, null_y], dim=0)
            
            u_combined = model(z_combined, r_combined, t_combined, y=y_combined)
            u_cond, u_uncond = u_combined.chunk(2, dim=0)
            
            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            u = model(latents, r, t, y=y)
        
        # x_0 = x_1 - u(x_1, 0, 1)
        x0 = latents - u
        
    else:
        z = latents
        
        time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            
            t = torch.full((batch_size,), t_cur, device=device)
            r = torch.full((batch_size,), t_next, device=device)
            
            if do_cfg:
                z_combined = torch.cat([z, z], dim=0)
                r_combined = torch.cat([r, r], dim=0)
                t_combined = torch.cat([t, t], dim=0)
                y_combined = torch.cat([y, null_y], dim=0)
                
                u_combined = model(z_combined, r_combined, t_combined, y=y_combined)
                u_cond, u_uncond = u_combined.chunk(2, dim=0)
                
                # Apply CFG
                u = u_uncond + cfg_scale * (u_cond - u_uncond)
            else:
                u = model(z, r, t, y=y)
            
            # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
            z = z - (t_cur - t_next) * u
        
        x0 = z
    
    return x0
