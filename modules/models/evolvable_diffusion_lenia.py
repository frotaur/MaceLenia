import torch, torch.nn, torch.nn.functional as F
import pygame
from nltk.downloader import update

from .lenia import MCLenia
import random

from .utils.leniaparams import LeniaParams
from .. import DiffusionLenia


class EvolvableDiffusionLenia(DiffusionLenia):
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
        self.set_init_circle()
        self.params = LeniaParams.arbi_gen(batch_size=self.batch, num_channels=self.C, device=self.device, k_size=31)
        self.update_params(self.params)
        self.counter = 0
        self.batch_mass = torch.zeros(self.batch, device=self.device)
        self.compute_batch_mass()
        self._temp = 10
        self.masses = [27000,27000]




    def step(self):
        super().step()
        self.keep_track()


    def keep_track(self):
        self.counter += 1
        if self.counter % 100 == 0 and self.counter != 0:
            self.set_init_circle()
            self.compute_batch_mass()
            mass_idxs = torch.argsort(self.batch_mass, descending=True)
            self.masses.append(self.batch_mass[mass_idxs[0]].item())
            self.params[:] = self.params[mass_idxs[0]]
            self.params[1:] = self.params[1:].mutate(magnitude=0.1, rate=0.1)
            self.update_params(self.params, k_size_override=None)

    def compute_batch_mass(self):
        self.batch_mass = self.state.view(self.batch,-1).sum(dim=1)

    def get_string_state(self):
        return f"total mass: {self.state.sum().item():.2f}, temp : {self.temp:.2f}, Showing Batch: {self.show_batch}, counter: {self.counter}, Best mass: {self.batch_mass.max().item():.2f}"