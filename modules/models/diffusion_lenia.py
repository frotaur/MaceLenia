import torch, torch.nn, torch.nn.functional as F
import pygame
from .lenia import MCLenia
import random

class DiffusionLenia(MCLenia):
    """
    Mass conserving Lenia-like Alife model
    """

    def __init__(
        self,
        size,
        dt,
        num_channels=3,
        params=None,
        state_init=None,
        device="cpu",
        has_food=False,
        interest_files=None,
        save_dir=".",
    ):
        """
        Args:
            size : tuple, (B,H,W) size of the automaton
            dt : float, time step size
            num_channels : int, number of channels
            params : dict, parameters of the automaton
            state_init : tensor, initial state of the automaton
            device : str, device to use
        """
        self.has_food = has_food # Needed for initialization

        super().__init__(
            size,
            dt,
            num_channels,
            params,
            state_init,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir,
        )

        self._temp = 1
        self.Aff = self.compute_affinity()
        
    @torch.no_grad()
    def step(self):
        """
        Steps the alife model by one time step
        """
        """B,C,H,W = self.state.shape
        Aff = self.compute_affinity()
    
        Z = F.pad(Aff, (1,1,1,1), mode='circular') # (B,C,H+2,W+2) for the (3,3) kernel
        Z = F.unfold(Z, kernel_size=(3,3)).reshape(B,C,9,H,W) # (B,C*9,H,W)
        Z = Z.sum(dim=2) # (B,C,H,W) local affinity normalization

        state_portions = self.state/Z
        state_portions = F.pad(state_portions, (1,1,1,1), mode='circular') # (B,C,H+2,W+2) for the (3,3) kernel
        state_portions = F.unfold(state_portions, kernel_size=(3,3)).reshape(B,C,9,H,W) # (B,C*H*W,9)
        self.state = (Aff[:,:,None]*state_portions).sum(dim=2) # (B,C,H,W) result of the diffusion"""
        B, C, H, W = self.state.shape

        Aff = self.compute_affinity()
        Aff_exp = F.pad(Aff, (1, 1, 1, 1), mode="circular")  # (B,C,H+2,W+2) for the (3,3) kernel
        Aff_exp = F.unfold(Aff_exp, kernel_size=(3, 3)).reshape(B, C, 9, H, W)  # (B,C*9,H,W)
        E = Aff_exp.sum(dim=2)
        E_exp = F.pad(E, (1, 1, 1, 1), mode="circular")
        E_exp = F.unfold(E_exp, kernel_size=(3, 3)).reshape(B, C, 9, H, W)  # (B,C*9,H,W)
        state_exp = F.pad(self.state, (1, 1, 1, 1), mode="circular")
        state_exp = F.unfold(state_exp, kernel_size=(3, 3)).reshape(B, C, 9, H, W)  # (B,C*9,H,W)

        self.state = ((Aff[:, :, None, ...] / E_exp) * state_exp).sum(dim=2)

        if self.has_food:
            """uncomment the death sections for death mechanics, but its finicky and i dont like it """
            where_food = self.food_channel > 0  # Where the food channels are
            where_contact = (
                self.state.sum(dim=1)[:, None, :, :] > 0.1
            )  # Where the eating channel is, we could amke this dynamic, 0.1 is the threshold for eating
            death = (
                (self.state.sum(dim=1)[:, None, :, :] < 0.01) & (self.state.sum(dim=1)[:, None, :, :] > 0)
            ) * self.state  # death of the feeding channel, very finicky

            overlap = where_food & where_contact  # where the channels overlap
            transfer = (
                torch.ones_like(where_food) * overlap * 0.03
            )  # How much to increase / deacrease the mass currently set to 0.01
            self.state += transfer  # Lenia mass increase
            self.state -= death
            self.food_channel -= transfer  # Food mass deacrease
            self.food_channel += death.sum(dim=1)[:, None, :, :] / 3

    def compute_affinity(self):
        """
        Computes the affinity matrix of the model
        """
        Aff = self.kernel_fftconv(self.state)  # (B,C,C,H,W) first step affinity, usual convolutions

        weights = self.weights[..., None, None]  # (B,C,C,1,1)
        Aff = (self.growth(Aff) * weights).sum(dim=1)  # (B,C,H,W) pre-exponential affinity
        Aff = torch.exp(self.temp * Aff)

        return Aff

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, value):
        self._temp = value

    def process_event(self, event, camera=None):
        """
        UP -> Increase temperature
        DOWN -> Decrease temperature
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.temp += 0.2
            if event.key == pygame.K_DOWN:
                self.temp -= 0.2

    process_event.__doc__ = MCLenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )  # Hack to append the docstring of MCLenia.process_event

    def get_string_state(self):
        return f"total mass: {self.state.sum().item():.2f}, temp : {self.temp:.2f}"
    
    def random_food_chan(self, num_spots=100, food_size=5):
        """
            Returns a food channel with num_spots of food of size food_size
            Args :
                num_spots : int, number of food spots
                food_size : int, size of the food spots
            
            Returns :
                food_chan : tensor, (B,1,H,W) food channel
        """
        places = [[random.randint(food_size, self.h - food_size), random.randint(food_size, self.w - food_size)] for _ in
                  range(num_spots)]
        food_chan = torch.zeros((self.batch, 1, self.h, self.w), device=self.device)
        for place in places:
            food_chan[:,:,place[0] - food_size: place[0] + food_size, place[1] - food_size:place[1] + food_size] = 1
        
        return food_chan

    def set_init_fractal(self):
        super().set_init_fractal()
        if self.has_food:
            self.food_channel = self.random_food_chan() # (B,1, H,W)

    def set_init_perlin(self, wavelength=None):
        super().set_init_perlin(wavelength)
        if self.has_food:
            self.food_channel = self.random_food_chan() # (B,1, H,W)

    def set_init_circle(self, fractal=False, radius=None):
        super().set_init_circle(fractal, radius)
        if self.has_food:
            self.food_channel = self.random_food_chan() # (B,1, H,W)

    @torch.no_grad()
    def draw(self):
        """
            Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"

        toshow= self.state[0].clone() # (C,H,W), pygame conversion done later

        if(self.C==1):
            toshow = toshow.repeat(3,1,1) # (3,H,W)
        elif(self.C==2):
            toshow = torch.cat([toshow,torch.zeros_like(toshow)],dim=0) # (3,H,W)
        else :
            toshow = toshow[:3,:,:] # (3,H,W)

        if self.has_food:
            toshow[:,:,:] += self.food_channel[0]# (1,H,W)

        if self.display_kernel == True:
            kern = self.compute_ker()  # (C,3,k_size,k_size)
            for i in range(kern.shape[0]):
                toshow[:, self.h - self.k_size : self.h, i * self.k_size : (i + 1) * self.k_size] = kern[
                    i
                ].cpu()

        self._worldmap= torch.clamp(toshow,0.,1.) 

