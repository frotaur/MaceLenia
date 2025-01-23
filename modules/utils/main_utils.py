import torch
 
def compute_ker(auto, device):
    """
        Prepares the kernel and translate it to an RGB image for viewing.
    """
    kern= auto.k[0].detach() # (C,k_mult*C, k_size, k_size), removed batch
    # print('Kern shape : ', kern.shape)
    # show_image(kern,rescale=True)
    if(kern.shape[1]==1):
        kern = kern.expand(1,3,-1,-1)
    elif(kern.shape[1]>3):
        kern = kern[:,:3] # Cut, and include only the first set of kernels
    kern = kern.permute((0,3,2,1)) # (C,k_size,k_size,3)
    maxs = torch.tensor((torch.max(kern[0]), torch.max(kern[1]), torch.max(kern[2])), device=device)
    # print(maxs)
    maxs = maxs[:,None,None,None]
    kern = kern/maxs 

    return kern # (C,k_size,k_size,3)


def create_smooth_circular_mask(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    H, W = tensor.shape[-2], tensor.shape[-1]
    center_y = (H - 1) / 2  # Allow fractional center for better smoothness
    center_x = (W - 1) / 2
    y = torch.linspace(0, H - 1, H, device=tensor.device).view(-1, 1)
    x = torch.linspace(0, W - 1, W, device=tensor.device).view(1, -1)
    distance = ((y - center_y) ** 2 + (x - center_x) ** 2).sqrt()
    smooth_transition = 0.5  # Define a region for the smooth transition (around the edge of the circle)
    mask = torch.clamp(1 - (distance - radius) / smooth_transition, 0, 1)
    masked_tensor = tensor * mask

    return masked_tensor