import torch
 
def compute_ker(auto, device):
    """
        Prepares the kernel and translate it to an RGB image for viewing.
    """
    kern= auto.compute_kernel()[0] # (C,k_mult*C, k_size, k_size), removed batch
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