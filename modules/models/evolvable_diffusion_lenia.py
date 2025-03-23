import torch, torch.nn, torch.nn.functional as F
import pygame
from nltk.downloader import update

from .lenia import MCLenia
import random

from .utils.leniaparams import LeniaParams
from .. import DiffusionLenia


class EvolvableDiffusionLenia(DiffusionLenia):
    """ An evolvable version of DiffusionLenia"""

    def __init__(
            self,
            size,
            dt,
            num_channels=3,
            params=None,
            state_init=None,
            device="cuda:0",
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


        super().__init__(
            size,
            dt,
            num_channels,
            params,
            state_init,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir,
            has_food=has_food
        )

        assert (self.batch % 2) == 0, "The batch size for evolution must be even"
        self.set_init_circle()
        self.base_state = self.state.clone()
        self.params = LeniaParams.arbi_gen(batch_size=self.batch, num_channels=self.C, device=self.device, k_size=31)
        self.update_params(self.params)
        self.counter = 0
        self.batch_mass = torch.zeros(self.batch, device=self.device)

        self._temp = 10
        self.masses = [27000,27000]
        self.base_food = self.food_channel.clone()
        self.split = self.batch//2




    def step(self):
        super().step()
        self.keep_track()


    def keep_track(self):
        self.counter += 1
        if self.counter % 1000 == 0 and self.counter != 0:

            p_idxs = self.tournament_selection(self.split)
            self.masses.append(self.batch_mass.mean().item())
            parents = self.params[p_idxs]
            self.params = self.p2p_crossover(parents, self.split, self.split)
            self.params[self.split:] = self.params[self.split:].mutate(magnitude=0.1, rate=0.1)
            self.update_params(self.params, k_size_override=None)

            #self.food_channel = self.base_food.clone()
            self.set_init_circle()




    def p2p_crossover(self, parent_params ,num_parents, num_children):

        new_pop = parent_params
        for i in range(num_children):
            child = parent_params[0]
            parents = random.sample(range(num_parents), 2)
            for key in parent_params.param_dict.keys():
                if (isinstance(parent_params[key], torch.Tensor)):
                    mask = torch.rand_like(parent_params[key][0]) > 0.5
                    child[key][0] = parent_params[key][parents[0]]* mask + parent_params[key][parents[1]]* (~ mask)
            new_pop = new_pop.cat(child)

        return new_pop

    def tournament_selection(self, num_winners):
        self.batch_mass = self.state.view(self.batch, -1).sum(dim=1)
        mass_idxs = torch.argsort(self.batch_mass, descending=True)
        return mass_idxs[:num_winners].tolist()

    def get_string_state(self):
        return f"total mass: {self.state.sum().item():.2f}, temp : {self.temp:.2f}, Showing Batch: {self.show_batch}, counter: {self.counter}, Best mass: {self.batch_mass.max().item():.2f}"