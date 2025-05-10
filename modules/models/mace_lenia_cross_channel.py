import torch, torch.nn, torch.nn.functional as F
import pygame
from nltk.downloader import update
from numpy.ma.core import minimum
from sympy.abc import alpha

from .mace_lenia import MaCELenia
from .utils.torch_utils import unfold3d
from .lenia import Lenia
import random


class MaCELeniaCrossChannel(MaCELenia):
    """
        MaCELenia with cross channel step
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

        super().__init__(
            size,
            dt,
            num_channels,
            params,
            state_init,
            has_food=has_food,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir,
        )

        self._beta = 6  # default temperature is 6 for this one
        self.alpha = 0.03
        self.params["alpha"] = self.alpha

    def update_params(self, params, k_size_override=None):
        super().update_params(params, k_size_override=k_size_override)
        if "alpha" in params:
            self.alpha = params["alpha"]

    def step(self, sense_food=False):
        """
        Steps the alife model by one time step
        """
        Aff = self._mace_step(sense_food=sense_food)
        if self.has_food:
            self._food_step()
        self._cross_chan_step(Aff)  # (B,C,H,W) cross channel step

    def _cross_chan_step(self, Aff):
        """Performs the cross channel step, given the affinity matrix"""
        max_Aff = torch.max(Aff, dim=1, keepdim=True)[0]
        Aff_shifted = self.b * (Aff - max_Aff)
        numerator = torch.exp(Aff_shifted)
        Aff_c = numerator / (numerator.sum(dim=1, keepdim=True))

        target_cross_c_masses = self.state.sum(dim=1, keepdim=True) * Aff_c
        self.state = self.state - (self.state - target_cross_c_masses) * self.alpha

    def process_event(self, event, camera=None):
        """
        LEFT/RIGHT: change alpha
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.alpha -= 0.02
                self.params["alpha"] = self.alpha
            if event.key == pygame.K_RIGHT:
                self.alpha += 0.02
                self.params["alpha"] = self.alpha

    process_event.__doc__ = Lenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )  # Hack to append the docstring of MCLenia.process_event

    def get_string_state(self):
        return super().get_string_state() + f"alpha: {self.alpha:.2f}"
