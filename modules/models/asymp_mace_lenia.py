from .mace_lenia import MaCELenia
import torch.nn.functional as F
import pygame, torch


class AsymptoticMaCELenia(MaCELenia):
    """
    Like diffusion lenia, but keeps track of dx, dt s.t. it has
    a well-defined continuous limit, both in t and x.

    For consistency, fixes k_size as a function of dx.
    NOTE : For now, does not support food, because I'm not sure
    about the food dynamics in the continuous limit.
    """

    def __init__(
        self,
        size,
        dx=0.08,
        num_channels=3,
        params=None,
        state_init=None,
        device="cpu",
        interest_files=None,
        save_dir=".",
    ):
        self.dx = dx
        self.dt = dx**2 / 3  # Maximum dt for stability

        super().__init__(
            size,
            self.dt,
            num_channels,
            params,
            state_init,
            device,
            has_food=False,
            sense_food=False,
            interest_files=interest_files,
            save_dir=save_dir,
        )

        print("k_size equivalent dx : ", self.compute_ksize())
        print(f"Set dt={self.dt:.3f}  for stability")

    def compute_ksize(self):
        k_size = int(2 / self.dx)
        k_size = k_size + 1 if k_size % 2 == 0 else k_size
        return k_size

    def update_params(self, params, k_size_override=None):
        # Update params, but force k_size to be consistent with dx
        if "dx" in params:
            self.dx = params["dx"]
        super().update_params(params, k_size_override=self.compute_ksize())

    def _mace_step(self, sense_food=False):
        """
        Performs the Mace Step of the model, including the dx and dt parameters
        and returns the affinity tensor for potential further use.

        Args:
            sense_food : bool, if True, the model will sense food
            and update the affinity tensor accordingly
        Returns:
            Aff : (B,C,H,W), affinity tensor of the model
        """
        B, C, H, W = self.state.shape
        print('dt is ; ', self.dt)
        # Compute affinity with growth function of Lenia
        Aff = self._compute_affinity(sense_food=sense_food)  # (B,C,H,W) affinity matrix
        
        expAff = torch.exp(self.b * Aff-self.cr/2)  # Exponentiate with beta

        # Unfold expAff to prepare the computation of normalization Z
        Z = F.pad(expAff, (1, 1, 1, 1), mode="circular")  # (B,C,H+2,W+2) for the (3,3) kernel
        Z = F.unfold(Z, kernel_size=(3, 3)).reshape(B, C, 9, H, W)  # (B,C*9,H,W)
        Z = Z.sum(dim=2)  # (B,C,H,W) local affinity normalization tensor

        to_give = self.state / Z  # normalized mass, ready to be portioned according to the expAff
        to_give = F.pad(to_give, (1, 1, 1, 1), mode="circular")  # (B,C,H+2,W+2) for the (3,3) kernel
        to_give = F.unfold(to_give, kernel_size=(3, 3)).reshape(
            B, C, 9, H, W
        )  # (B,C,9,H,W), unfold again to distribute mass to all 9 neighbors

        redistribution = (expAff[:, :, None] * to_give).sum(dim=2)  # (B,C,H,W) result of the diffusion

        self.state = self.state + 3 * self.dt / (self.dx**2) * (redistribution - self.state)

        return Aff  # (B,C,H,W), return affinity if needed later (e.g. for cross channel step)

    def process_event(self, event, camera=None):
        """
        Wheel -> Change dx, while keeping dt/dx^2 constant
        Shift + Wheel -> Change dx only. Might become unstable
        Alt + Wheel -> Change dt only. Might become unstable
        """
        super().process_event(event, camera)
        if event.type == pygame.MOUSEWHEEL and not (pygame.key.get_mods() & pygame.KMOD_CTRL):
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                # Change dx with shift + scroll
                if event.y > 0:
                    self.dx *= 1.1
                else:
                    self.dx *= 0.9
                self.update_params(self.params)
            elif pygame.key.get_mods() & pygame.KMOD_ALT:
                # Change dt with scroll + Alt
                if event.y > 0:
                    self.dt *= 1.1
                else:
                    self.dt *= 0.9
            else:
                # Change dx while keeping dt/dx^2 constant when just scrolling
                factor = self.dt / (self.dx**2)
                if event.y > 0:
                    self.dx *= 1.1
                else:
                    self.dx *= 0.9
                self.dt = factor * self.dx**2
                self.update_params(self.params)

    process_event.__doc__ = MaCELenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip("\n")

    def get_string_state(self):
        return (
            super().get_string_state()
            + f" dx={self.dx * 1000:.2f}, dt={self.dt * 100:.2f}/100, factor={3 * self.dt / (self.dx**2):.2f}"
        )
