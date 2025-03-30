import torch, torch.nn, torch.nn.functional as F
import pygame
from nltk.downloader import update

from .diffusion_lenia_cross_channel import DiffusionLeniaCrossChannel
from .lenia import MCLenia
import random

from .utils.leniaparams import LeniaParams, BatchParams
from .. import DiffusionLenia


class EvolvableDiffusionLenia(DiffusionLeniaCrossChannel):
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
        self.state[:,...] = self.state[0,...]
        self.params = LeniaParams.default_gen(batch_size=self.batch, num_channels=self.C, device=self.device, k_size=31)
        self.update_params(self.params)
        self.counter = 0
        self.batch_mass = torch.zeros(self.batch, device=self.device)

        self._temp = 10
        self.masses = [8000,8000]
        self.base_state = self.state.clone()
        self.base_food = self.food_channel.clone()
        self.mutation_params = self.get_es_params()
        self.split = self.batch//2



    def get_es_params(self) -> dict:
        params = {}
        for k,v in self.params.param_dict.items():
            if isinstance(v, torch.Tensor):
                params[k] = torch.rand((self.batch), device=self.device)

        return params

    def step(self):
        self.keep_track()
        super().step(sense_food=True)



    def keep_track(self):
        self.counter += 1
        if self.counter % 1000 == 0 and self.counter != 0:

            p_idxs = self.tournament_selection(self.split, some_diversity=True)

            self.masses.append(self.batch_mass.max().item())
            parents = self.params[p_idxs]
            parent_mutation_params = {k:params[p_idxs] for k,params in self.mutation_params.items()}
            self.params, self.mutation_params = self.p2p_crossover(parents, parent_mutation_params ,self.split, self.split)
            self.params[self.split:] = self.mutate(rate = 0.05, magnitude=0.05, params=self.params[self.split:])
            self.update_params(self.params, k_size_override=None)
            self.set_init_circle()
            #self.state = self.base_state.clone()
            #self.food_channel = self.base_food.clone()




    def p2p_crossover(self, parent_params: LeniaParams, parent_mutation_params: dict ,num_parents: int, num_children: int) -> (LeniaParams, dict):

        new_pop = parent_params
        for i in range(num_children):
            child = {}
            parents = random.sample(range(num_parents), 2)
            for key in parent_params.param_dict.keys():

                if (isinstance(parent_params[key], torch.Tensor)):

                    mask = torch.rand_like(parent_params[key][0]) >= 0.5
                    mask_mut = torch.rand_like(parent_mutation_params[key][0]) < 0.5
                    new_mut = (parent_mutation_params[key][parents[0]:parents[0]+1].clone()* mask_mut).float() + (parent_mutation_params[key][parents[1]:parents[1]+1].clone()* (~ mask_mut)).float()
                    parent_mutation_params[key] = torch.cat((parent_mutation_params[key], new_mut), dim=0)
                    child[key] = (parent_params[parents[0]:parents[0]+1][key].clone()* mask).float() + (parent_params[parents[1]:parents[1]+1][key].clone()* (~ mask)).float()
                else :

                    child[key] = parent_params[key]

            new_pop = new_pop.cat(LeniaParams(param_dict=child, device=self.device))

        return new_pop, parent_mutation_params

    def tournament_selection(self, num_winners: int, some_diversity: bool = False) -> list[int]:
        self.batch_mass = self.state.view(self.batch, -1).sum(dim=1)
        mass_idxs = torch.argsort(self.batch_mass, descending=True, stable=True)
        if some_diversity:
            best_idxs = mass_idxs[:num_winners-1].tolist()
            diversity = mass_idxs[-(self.batch//4)].tolist()
            best_idxs.append(diversity)
        else:
            best_idxs = mass_idxs[:num_winners].tolist()

        return best_idxs


    def mutate(self, rate: float, magnitude:float, params: LeniaParams) -> LeniaParams:
        new_params = {}
        n_0_1 = torch.randn((self.batch), device=self.device )
        for key in params.param_dict.keys():
            if isinstance(params[key], torch.Tensor):
                self.mutation_params[key] = self.mutation_params[key] * torch.exp(n_0_1 - torch.randn_like(self.mutation_params[key]))
                mask = torch.rand_like(params[key]) < rate

                new_params[key] = params[key].clone() + torch.randn_like(params[key]) * magnitude * mask * torch.sqrt(self.mutation_params[key][:self.split].view(*params[key].shape[:1], *([1] * (params[key].dim() - 1))))

            else : new_params[key] = params[key]
        return LeniaParams(param_dict=new_params, device=self.device)

    def get_string_state(self):
        return f"total mass: {self.state[self.show_batch].sum().item() + self.food_channel[self.show_batch].sum().item():.2f}, temp : {self.temp:.2f}, Showing Batch: {self.show_batch}, counter: {self.counter}, Best mass: {self.batch_mass.max().item():.2f}"