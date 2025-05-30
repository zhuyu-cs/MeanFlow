import torch

@torch.no_grad()
def meanflow_sampler(
    model, 
    latents,
    cfg_scale=1.0,
    num_steps=1, 
    **kwargs
):
    """
    MeanFlow sampler supporting both single-step and multi-step generation
    """
    batch_size = latents.shape[0]
    device = latents.device
    
    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        u = model(latents, noise_labels_r=r, noise_labels_t=t)
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

            u = model(z, noise_labels_r=r, noise_labels_t=t)
            
            # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
            z = z - (t_cur - t_next) * u
        
        x0 = z
    
    return x0