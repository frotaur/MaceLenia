import torch, torch.nn.functional as F
import pygame
from nltk.downloader import update
import random
from .utils.leniaparams import LeniaParams
from .mace_lenia import MaCELenia
import itertools
import math


class EvolvableMaCELenia(MaCELenia):
    """An evolvable version of MaCELenia"""

    def __init__(
        self,
        size,
        dt,
        num_channels=3,
        params=None,
        state_init=None,
        device="cuda:0",
        has_food=False,
        sense_food=False,
        interest_files=None,
        save_dir=".",
    ):
        super().__init__(
            size,
            dt,
            num_channels,
            params,
            state_init,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir,
            has_food=has_food,
            sense_food=sense_food,
        )

        assert ((self.batch % 2) == 0) and ((math.sqrt(self.batch)).is_integer()), (
            "The batch size for evolution must be even and a square number (e.g. 4,16, 36) "
        )
        self.set_init_circle()
        self.state[:, ...] = self.state[0, ...]
        self.params = LeniaParams.default_gen(
            batch_size=self.batch, num_channels=self.C, device=self.device, k_size=31
        )
        self.update_params(self.params)
        self.counter = 0
        self.batch_mass = torch.zeros(self.batch, device=self.device)

        self._beta = 10
        self.masses = [8000, 8000]
        self.base_state = self.state.clone()

        self.base_food = self.random_food_chan(food_amount=self.initial_food)
        self.mutation_params = self.get_es_params()
        self.split = self.batch // 2
        self.show_all = False
        self.manual_evolution = False
        self.mm = 0.05
        self.mr = 0.05

    def get_es_params(self) -> dict:
        params = {}
        for k, v in self.params.param_dict.items():
            if isinstance(v, torch.Tensor):
                params[k] = torch.rand((self.batch), device=self.device)

        return params

    def step(self):
        super().step()
        if not self.manual_evolution:
            self.keep_track()

    def keep_track(self):
        self.counter += 1
        if self.counter % 1000 == 0 and self.counter != 0:
            p_idxs = self.tournament_selection(self.split, some_diversity=True)

            self.masses.append(self.batch_mass.max().item())
            parents = self.params[p_idxs]
            parent_mutation_params = {k: params[p_idxs] for k, params in self.mutation_params.items()}
            self.params, self.mutation_params = self.p2p_crossover(
                parents, parent_mutation_params, self.split, self.split
            )
            self.params[self.split :] = self.mutate(
                rate=self.mr, magnitude=self.mm, params=self.params[self.split :]
            )
            self.update_params(self.params, k_size_override=None)
            self.set_init_circle()
            # self.state = self.base_state.clone()
            # self.food_channel = self.base_food.clone()

    def p2p_crossover(
        self, parent_params: LeniaParams, parent_mutation_params: dict, num_parents: int, num_children: int
    ) -> tuple[LeniaParams, dict]:
        new_pop = parent_params
        for i in range(num_children):
            child = {}
            parents = random.sample(range(num_parents), 2)
            for key in parent_params.param_dict.keys():
                if isinstance(parent_params[key], torch.Tensor):
                    mask = torch.rand_like(parent_params[key][0]) >= 0.5
                    mask_mut = torch.rand_like(parent_mutation_params[key][0]) < 0.5
                    new_mut = (
                        parent_mutation_params[key][parents[0] : parents[0] + 1].clone() * mask_mut
                    ).float() + (
                        parent_mutation_params[key][parents[1] : parents[1] + 1].clone() * (~mask_mut)
                    ).float()
                    parent_mutation_params[key] = torch.cat((parent_mutation_params[key], new_mut), dim=0)
                    child[key] = (parent_params[parents[0] : parents[0] + 1][key].clone() * mask).float() + (
                        parent_params[parents[1] : parents[1] + 1][key].clone() * (~mask)
                    ).float()
                else:
                    child[key] = parent_params[key]

            new_pop = new_pop.cat(LeniaParams(param_dict=child, device=self.device))

        return new_pop, parent_mutation_params

    def channel_mass_diff(self) -> torch.Tensor:
        indices = range(self.C)

        combination_tuples = itertools.combinations(indices, 2)
        sum_tensor = torch.zeros(self.batch, device=self.device)
        for pair in combination_tuples:
            sum_tensor += torch.abs(
                self.state[pair[0]].view(self.batch, -1).sum(dim=1)
                - self.state[pair[1]].view(self.batch, -1).sum(dim=1)
            )

        return sum_tensor

    def tournament_selection(self, num_winners: int, some_diversity: bool = False) -> list[int]:
        self.batch_mass = self.state.view(self.batch, -1).sum(dim=1)

        # self.batch_mass = torch.nan_to_num(self.batch_mass, nan=-10000000.0)
        mass_idxs = torch.argsort(self.batch_mass, descending=True)
        if some_diversity:
            best_idxs = mass_idxs[: num_winners - 1].tolist()
            diversity = mass_idxs[-(self.batch // 4)].tolist()
            best_idxs.append(diversity)
        else:
            best_idxs = mass_idxs[:num_winners].tolist()

        return best_idxs

    def mutate(
        self, rate: float, magnitude: float, params: LeniaParams, p_idxs: list[int] = None
    ) -> LeniaParams:
        new_params = {}
        n_0_1 = torch.randn((self.batch), device=self.device)
        for key in params.param_dict.keys():
            if isinstance(params[key], torch.Tensor):
                self.mutation_params[key] = self.mutation_params[key] * torch.exp(
                    n_0_1 - torch.randn_like(self.mutation_params[key])
                )
                mask = torch.rand_like(params[key]) < rate
                if p_idxs is None:
                    new_params[key] = params[key].clone() + torch.randn_like(
                        params[key]
                    ) * magnitude * mask * torch.sqrt(
                        self.mutation_params[key][: self.split].view(
                            *params[key].shape[:1], *([1] * (params[key].dim() - 1))
                        )
                    )
                else:
                    new_params[key] = params[key].clone() + torch.randn_like(
                        params[key]
                    ) * magnitude * mask * torch.sqrt(
                        self.mutation_params[key][p_idxs].view(
                            *params[key].shape[:1], *([1] * (params[key].dim() - 1))
                        )
                    )

            else:
                new_params[key] = params[key]
        return LeniaParams(param_dict=new_params, device=self.device)

    def get_string_state(self):
        return f"total mass: {self.state[self.show_batch].sum().item():.2f}, beta : {self.b:.2f}, Showing Batch: {self.show_batch}, counter: {self.counter}, Best mass: {self.batch_mass.max().item():.2f}, Manual Evolution: {self.manual_evolution}"

    def manual_evolution_trigger(self, camera=None):
        keys = self.get_mouse_state(camera=camera)
        if keys["left"]:
            adjusted_h = self.size[0] * 2
            adjusted_w = self.size[1] * 2
            click_x = keys["x"]
            click_y = keys["y"]
            subdivisions_per_side = math.sqrt(self.batch)
            col_index = int(click_x * subdivisions_per_side // adjusted_h)
            row_index = int(click_y * subdivisions_per_side // adjusted_w)
            quadrant_index = int((row_index * subdivisions_per_side) + col_index)
            print(quadrant_index)
            parent = self.params[quadrant_index]
            self.params[:] = parent
            for k, v in self.mutation_params.items():
                self.mutation_params[k][:, ...] = self.mutation_params[k][quadrant_index, ...]

            non_parent_ids = [i for i in range(self.batch) if i != quadrant_index]

            self.params[non_parent_ids] = self.mutate(
                rate=self.mr, magnitude=self.mm, params=self.params[non_parent_ids], p_idxs=non_parent_ids
            )
            self.update_params(self.params, k_size_override=None)
            self.set_init_circle()

    def process_event(self, event, camera=None):
        """
        LMB -> In manual Evolution, Select Params to mutate
        V - > Toggles manual evolution
        """
        super().process_event(event, camera)

        if (event.type == pygame.MOUSEBUTTONDOWN) and self.manual_evolution:
            mods = pygame.key.get_mods()
            if not (mods & pygame.KMOD_LCTRL):
                self.manual_evolution_trigger(camera=camera)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                self.manual_evolution = not self.manual_evolution
                self.show_all_override = self.manual_evolution

    process_event.__doc__ = MaCELenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )  # Hack to append the docstring of MCLenia.process_event
