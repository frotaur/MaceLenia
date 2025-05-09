import torch, torch.nn, torch.nn.functional as F
import pygame
from nltk.downloader import update
from numpy.ma.core import minimum
import showtens
from .lenia import MCLenia
import random
import math, time

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
        sense_food=False,
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
        self.initial_food = 10000
        self.sense_food = sense_food
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

        self._temp = 8
        self.Aff = self.compute_affinity()
        self.show_batch = 0
        self.cum_loss_mass = torch.zeros(self.batch, device=device)
        self.show_all = False
        self.show_all_override = False
        
        kernel_size = 7
        self.smear_kernel = torch.ones((1,1,kernel_size,kernel_size), device=self.device)/(kernel_size*kernel_size) # (1,1,kernel_size,kernel_size)


    def step(self, sense_food = None):
        """
        Steps the alife model by one time step

        Args: sense_food: overrides self.sense_food, if True, the model will sense food
        """
        if sense_food is not None:
            self._mace_step(sense_food=sense_food)
        else:
            self._mace_step(sense_food=self.sense_food)

        if self.has_food:
            self._food_step()

    def _mace_step(self,sense_food = False):
        """
            Performs the Mace Step of the model,
            and returns the affinity tensor

            Args:
                sense_food : bool, if True, the model will sense food
                and update the affinity tensor accordingly
            Returns:
                Aff : tensor, affinity tensor of the model
        """
        B,C,H,W = self.state.shape

        Aff = self.compute_affinity(sense_food=sense_food) # (B,C,H,W) affinity matrix
        expAff = torch.exp(self.temp * Aff)

        Z = F.pad(expAff, (1,1,1,1), mode='circular') # (B,C,H+2,W+2) for the (3,3) kernel
        Z = F.unfold(Z, kernel_size=(3,3)).reshape(B,C,9,H,W) # (B,C*9,H,W)
        Z = Z.sum(dim=2) # (B,C,H,W) local affinity normalization

        state_portions = self.state/Z
        state_portions = F.pad(state_portions, (1,1,1,1), mode='circular') # (B,C,H+2,W+2) for the (3,3) kernel
        state_portions = F.unfold(state_portions, kernel_size=(3,3)).reshape(B,C,9,H,W) # (B,C,9,H,W)
        self.state = (expAff[:,:,None]*state_portions).sum(dim=2) # (B,C,H,W) result of the diffusion

        return Aff

    def _food_step(self):
        if self.has_food:
            food_amount = 200
            # Decay proportionally to mass, but with a minimum rate
            # alowable_decay = torch.minimum(self.state, self.state*0.003 + torch.full_like(self.state, 0.0003))
            # alowable_decay = torch.where(self.state>0.01,self.state*0.0007+torch.full_like(self.state,0.02*0.0005), torch.zeros_like(self.state))
            
            allowable_decay = torch.where(self.state>0.005,0.0003, torch.zeros_like(self.state))
            self.state = (self.state  - allowable_decay)  # Update the state by subtracting the allowable decay
            # Compute all mass lost, when above some threshold, reintroduce the mass as food
            self.cum_loss_mass+= (allowable_decay).view(self.batch, -1).sum(dim = 1)
            update_idxs = torch.argwhere(self.cum_loss_mass >= food_amount).tolist()
            update_idxs = [p[0] for p in update_idxs]

            if update_idxs:
                self.cum_loss_mass[update_idxs] = self.cum_loss_mass[update_idxs] - food_amount
                self.food_channel = self.random_food_chan(food_amount=food_amount,num_spots=1,food_size=7,add_to_exisitng=True,channels=update_idxs)

            self.update_food(min_density = 0.05,transfer_rate=0.06, death_enabled=False)

    def update_food(self, min_density=0.1, transfer_rate=0.03, death_enabled=False):
        """uncomment the death sections for death mechanics, but its finicky and i dont like it """
        where_food = self.food_channel > 0  # Where the food channels are
        where_contact = (self.state.sum(dim=1,keepdim=True) >= min_density)  # Where the eating channel is, we could amke this dynamic, 0.1 is the threshold for eating

        if death_enabled:
            death = ((self.state < 0.04) & (self.state > 0)) * self.state  # death of the feeding channel, very finicky

        overlap = where_food & where_contact  # where the channels overlap
        transfer = torch.minimum(self.food_channel, torch.ones_like(where_food) * overlap * transfer_rate)

        self.state += transfer/3  # Lenia mass increase


        self.food_channel -= transfer


        if death_enabled:
            self.state -= death
            self.food_channel += death.sum(dim=1, keepdim=True)

    def update_show_batch(self, dirr):
        self.show_batch = (self.show_batch + dirr) % self.batch



    def compute_affinity(self, sense_food = False):
        """
        Computes the affinity matrix of the model
        """
        # if sense_food and self.has_food:
            # a = self.state.clone()
            # a[:,0:1,...] += self.food_channel # Add food channel to the state 'r' channel, for sensing
            # Aff = self.kernel_fftconv(a)  # (B,C,C,H,W) first step affinity, usual convolutions
        # else:
        #     Aff = self.kernel_fftconv(self.state)

        Aff = self.kernel_fftconv(self.state)
        weights = self.weights[..., None, None]  # (B,C,C,1,1)
        Aff = (self.growth(Aff) * weights).sum(dim=1)  # (B,C,H,W) pre-exponential affinity
        
        if(self.has_food and sense_food):
            food_aff = (self.food_channel>0).float().expand(-1, self.C, -1, -1) # (B,1,H,W) food channel
            # Optional : increase affinity according also to how much food is sensed
            food_aff = food_aff + self.kernel_fftconv(food_aff).sum(dim=1) # (B,1,H,W) food channel
            # Hardcoded for now, but remove affinity when matter is too low, so it cant eat
            Aff = Aff + (food_aff)#*(self.state>0.05) # (B,C,H,W) food affinity

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
        PLUS -> Show next batch
        MINUS -> Show previous batch
        B -> Toggle show all batches at once
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.temp += 0.2
            if event.key == pygame.K_DOWN:
                self.temp -= 0.2
            if event.key == pygame.K_KP_PLUS or event.key == pygame.K_PLUS:
                self.update_show_batch(1)
            if event.key == pygame.K_KP_MINUS or event.key == pygame.K_MINUS:
                self.update_show_batch(-1)
            if event.key == pygame.K_b:
                self.show_all = not self.show_all

        mouse_state = self.get_mouse_state(camera)
        if(mouse_state.left or mouse_state.right):
            add_rad = self.k_size/2.
            x, y = mouse_state.x, mouse_state.y
            add_mask = (self.X-x)**2 + (self.Y-y)**2 < add_rad**2  # (H,W)

            if(mouse_state.left):
                addition = torch.rand((self.batch, self.C, self.h, self.w), device=self.device)
                self.state[:,:, add_mask] += 0.05*addition[:, :, add_mask]
            elif(mouse_state.right):
                self.state[:,:, add_mask] -= 0.05
                self.state[:,:, add_mask] = torch.clamp(self.state[:,:, add_mask], 0., 1.)



    process_event.__doc__ = MCLenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )  # Hack to append the docstring of MCLenia.process_event

    def get_string_state(self):
        return super().get_string_state()+f"tot mass: {self.total_mass():.2f}, live: {self.state.sum():.2f} temp : {self.temp:.2f}, Showing Batch: {self.show_batch}"
    
    def random_food_chan(self, food_amount, num_spots=300, food_size=5, add_to_exisitng = False, channels= []):
        """
            Returns a food channel with num_spots of food of size food_size
            Args :
                num_spots : int, number of food spots
                food_size : int, size of the food spots
            
            Returns :
                food_chan : tensor, (B,1,H,W) food channel
        """
        food_density = food_amount/(num_spots*food_size*food_size) # food_size*food_size is the area of the food spot
        places = [[random.randint(food_size, self.h - food_size), random.randint(food_size, self.w - food_size)] for _ in
                  range(num_spots)]

        if add_to_exisitng:
            food_chan = self.food_channel.clone()
        else:
            food_chan = torch.zeros((self.batch, 1, self.h, self.w), device=self.device)
        for place in places:
            if add_to_exisitng:
                food_chan[channels, :, place[0] - food_size//2: place[0] + food_size//2+1,
                place[1] - food_size//2:place[1] + food_size//2+1] = food_density
            else:
                food_chan[:,:,place[0] - food_size//2: place[0] + food_size//2+1, place[1] - food_size//2:place[1] + food_size//2+1] = food_density
        
        return food_chan



    def set_init_fractal(self):
        super().set_init_fractal()
        if self.has_food:
            self.food_channel = self.random_food_chan(food_amount=self.initial_food) # (B,1, H,W)
            self.cum_loss_mass = torch.zeros(self.batch, device=self.device)

    def set_init_perlin(self, wavelength=None):
        super().set_init_perlin(wavelength)
        if self.has_food:
            self.food_channel = self.random_food_chan(food_amount=self.initial_food) # (B,1, H,W)
            self.cum_loss_mass = torch.zeros(self.batch, device=self.device)

    def set_init_circle(self, fractal=False, radius=None):
        super().set_init_circle(fractal, radius)
        if self.has_food:
            self.food_channel = self.random_food_chan(food_amount=self.initial_food) # (B,1, H,W)
            self.cum_loss_mass = torch.zeros(self.batch, device=self.device)



    @torch.no_grad()
    def draw(self):
        """
            Draws the RGB worldmap from state.
        """
        # assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        if (not self.show_all) and (not self.show_all_override)  :
            toshow = self.state[self.show_batch].clone()  # (C,H,W), pygame conversion done later

            if (self.C == 1):
                toshow = toshow.repeat(3, 1, 1)  # (3,H,W)
            elif (self.C == 2):
                toshow = torch.cat([toshow, torch.zeros_like(toshow)], dim=0)  # (3,H,W)
            else:
                toshow = toshow[:3, :, :]  # (3,H,W)

            if self.has_food:
                toshow[:, :, :] += self.food_channel[self.show_batch]  # (1,H,W)

            if self.display_kernel:
                toshow = self._draw_kernel(toshow)

            self._worldmap = torch.clamp(toshow, 0., 1.)


            # display grayscale where the kernel is big
            # conv = self.kernel_fftconv(self.state)  # (B,C,C,H,W)
            # whitepix = torch.any(torch.any((conv[0] > 2.0),dim=0),dim=0)  # (H,W)
            # self._worldmap[:,whitepix] = 0.4  # (3,H,W)
            # superwhite = torch.any(torch.any((conv[0] > 3.0),dim=0),dim=0)  # (H,W)
            # self._worldmap[:,superwhite] = 0.6
            # supersuperwhite = torch.any(torch.any((conv[0] > 4.0),dim=0),dim=0)  # (H,W)
            # self._worldmap[:,supersuperwhite] = 0.8
            # superduperwhite = torch.any(torch.any((conv[0] > 5.0),dim=0),dim=0)  # (H,W)
            # self._worldmap[:,superduperwhite] = 1.0


        else:
            mod_state = self.state.clone()
            mod_state[:, :, :, 0:5] = 1
            mod_state[:, :, :, -5:] = 1
            mod_state[:, :, 0:5, :] = 1
            mod_state[:, :, -5:, :] = 1

            if self.display_kernel == True:
                for j in range(self.batch):
                    kern = self.compute_ker(batch=j)  # (C,3,k_size,k_size)
                    for i in range(kern.shape[0]): # TODO : bugs out if num_channels =1 
                        mod_state[j,:, self.h - self.k_size: self.h, i * self.k_size: (i + 1) * self.k_size] = kern[
                            i
                        ].cpu()



            toshow = showtens.gridify(mod_state, max_width=self.size[1]*2, columns=int(math.sqrt(self.batch)))
            if (self.C == 1):
                toshow = toshow.repeat(3, 1, 1)  # (3,H,W)q
            elif (self.C == 2):
                toshow = torch.cat([toshow, torch.zeros_like(toshow)], dim=0)  # (3,H,W)
            else:
                toshow = toshow[:3, :, :]  # (3,H,W)

            if self.has_food:
                toshow[:, :, :] += showtens.gridify(self.food_channel, max_width=self.size[1]*2, columns= int(math.sqrt(self.batch)))  # (1,H,W)

            self._worldmap = torch.clamp(toshow, 0., 1.)

    def total_mass(self):
        """
        Returns the total mass of the model
        """
        if(self.has_food):
            return (self.state.sum(dim=(1, 2, 3)) + self.food_channel.sum(dim=(1, 2, 3)))[0]
        else:
            return self.state.sum(dim=(1, 2, 3))[0]