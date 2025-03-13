from.diffusion_lenia import DiffusionLenia
import torch.nn.functional as F
import pygame


class AsymptoticDiffusionLenia(DiffusionLenia):
    """ 
    Like diffusion lenia, but keeps track of dx, dt s.t. it has
    a well-defined continuous limit, both in t and x.

    For consistency, fixes k_size as a function of dx.
    NOTE : For now, does not support food, because I'm not sure
    about the food dynamics in the continuous limit.
    """

    def __init__(self, size, dt=0.01/3., dx=0.1, num_channels=3, params=None, state_init=None, device="cpu", interest_files=None, save_dir="."):
        self.dx = dx
        super().__init__(size,dt,num_channels,params,state_init,device,False,interest_files,save_dir)


        if (dt/(3*dx**2)<=1.):
            print(f"Ensure dt/(3*dx**2)<=1 for stability, currently {dt/(3*dx**2)}")

    def compute_ksize(self):
        k_size = int(2/self.dx)
        k_size = k_size + 1 if k_size % 2 == 0 else k_size
        return k_size

    def update_params(self, params, k_size_override=None):
        # Update params, but force k_size to be consistent with dx
        if 'dx' in params:
            self.dx = params['dx']
        super().update_params(params, k_size_override=self.compute_ksize())
    
    def step(self):
        """
        Steps the alife model by one time step
        """
        B,C,H,W = self.state.shape
        Aff = self.compute_affinity()
    
        Z = F.pad(Aff, (1,1,1,1), mode='circular') # (B,C,H+2,W+2) for the (3,3) kernel
        Z = F.unfold(Z, kernel_size=(3,3)).reshape(B,C,9,H,W) # (B,C*9,H,W)
        Z = Z.sum(dim=2) # (B,C,H,W) local affinity normalization

        state_portions = self.state/Z
        state_portions = F.pad(state_portions, (1,1,1,1), mode='circular') # (B,C,H+2,W+2) for the (3,3) kernel
        state_portions = F.unfold(state_portions, kernel_size=(3,3)).reshape(B,C,9,H,W) # (B,C*H*W,9)
        
        redistribution = (Aff[:,:,None]*state_portions).sum(dim=2) # (B,C,H,W) result of the diffusion

        self.state = self.state + 3*self.dt/(self.dx**2)*(redistribution - self.state)


    def process_event(self, event, camera=None):
        """
            Wheel -> Change dt
            Shift + Wheel -> Change dx
            Alt + Wheel -> Change dx while keeping dt/dx^2 constant
        """
        super().process_event(event, camera)
        if event.type == pygame.MOUSEWHEEL and not (pygame.key.get_mods() & pygame.KMOD_CTRL):
            if pygame.key.get_mods() & pygame.KMOD_ALT:
                # Change dx while keeping dt/dx^2 constant when holding Alt
                factor = self.dt/(self.dx**2)
                if event.y > 0:
                    self.dx *= 1.1
                else:
                    self.dx *= 0.9
                self.dt = factor * self.dx**2
                self.update_params(self.params)
            elif pygame.key.get_mods() & pygame.KMOD_SHIFT:
                # Change dx with shift + scroll
                if event.y > 0:
                    self.dx *= 1.1
                else:
                    self.dx *= 0.9
                self.update_params(self.params)
            else:
                # Change dt with scroll
                if event.y > 0:
                    self.dt *= 1.1
                else:
                    self.dt *= 0.9
                
    process_event.__doc__ = DiffusionLenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )
    def get_string_state(self):
        return super().get_string_state() + f"dx={self.dx:.3f}, dt={self.dt:.3f}, factor={3*self.dt/(self.dx**2):.2f}"