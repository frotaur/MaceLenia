import torch.nn as nn, torch
from torch import Tensor
import random
from torchenhanced import DevModule
from collections.abc import Callable


class ArbitraryFunction(nn.Module):
    """
    Batched Arbitrary Function Generator. It is agnostic to a specific shape of input,
    just expect a tensor of shape (N,*), where N is a 'batch size', and * is the rest of the dimensions,
    where the function is evaluated in. All arbitrary function of the batch will have the same number of harmonics.
    """

    def __init__(
        self,
        func_num: int = None,
        num_harmonics: int = None,
        coefficients: Tensor = None,
        harmonics: Tensor = None,
        shifts: Tensor = torch.tensor([0]),
        period: Tensor = torch.tensor([2]),
        bounds_range: tuple = (0, 1),
        clips_min = 0.0,
        device="cpu",
    ):
        """
        If coefficient and harmonics are not provided, they are randomly generated. Both of them
        must be provided if one is, otherwise it will raise an error. They will override the num_harmonics
        and func_num values if provided.
        Args:
            func_num: int, number of functions to generate. Acts like a 'batch' dimension
            num_harmonics : int, number of harmonics to use for each function
            coefficients: optional, (func_num, num_harmonics) tensor of coefficients to use
            harmonics: optional, (func_num, num_harmonics) tensor of harmonics to use
            bounds_range: tuple, (min,max) range of the function values
            device : torch device
        """
        super().__init__()
        self.device = device

        if coefficients is None and harmonics is None:
            assert func_num is not None and num_harmonics is not None, (
                "If coefficients and harmonics are not provided, func_num and num_harmonics must be"
            )
            self.func_num = func_num
            self.num_harmonics = num_harmonics
            # Randomly create harmonics and coeffs. Maybe in the future, add more parameters to sample harmonics ?
            coefficients = torch.randn(
                func_num, num_harmonics, device=device
            )  # Random, possibly negative, coefficients
            harmonics = torch.tensor(range(1, num_harmonics + 1), dtype=float, device=device)[None].expand(
                func_num, num_harmonics
            )
            harmonics = harmonics + torch.randn_like(harmonics)  # Add some variance to the harmonics values
        elif coefficients is not None and harmonics is not None:
            assert coefficients.shape == harmonics.shape, (
                "Coefficients and harmonics must have the same shape"
            )
            self.func_num = coefficients.shape[0]
            self.num_harmonics = coefficients.shape[1]
        else:
            raise ValueError("If coefficients or harmonics are provided, both must be provided")

        self.bounds_range = bounds_range
        self.clips_min = clips_min
        self.shifts = shifts.to(self.device)
        self.period = period.to(self.device)
        # Set the coefficients and harmonics as buffers
        # Change the following to nn.Parameter if we want to train the coefficients
        self.register_buffer("coefficients", coefficients)  # (B,num_harmonics)
        self.register_buffer("harmonics", harmonics)  # (B,num_harmonics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given x, evaluates the arbitrary function at the locations determined by x

        Args:
            x : Tensor of shape (N,*). B must match self.func_num
        """
        B, extra_dims = x.shape[0], x.shape[1:]

        assert x.shape[0] == self.func_num, "Dim 0 of input must match self.func_num of coefficients"


        x = x.reshape(self.func_num, 1, -1)  # (B,1,*)
        shift = x.min().unsqueeze(-1)
        period = 1/1000
        #values = self.coefficients[..., None] * torch.cos(
            #self.harmonics[..., None] * x * torch.pi
        #)  # (B,num_harmonics,*)


        values = torch.sum(
        self.coefficients[..., None] * torch.exp(1j * 2 * self.harmonics[..., None] * torch.pi * (x - self.shifts) / self.period),
        dim=1
    ).real
        #values = values.sum(dim=1, keepdim=False)  # (B,*), sum over harmonics

        min_vals, _ = values.min(dim=-1, keepdim=True)  # (B,1)
        max_vals, _ = values.max(dim=-1, keepdim=True)  # (B,1)

        normalized_tensor = (values - min_vals) / (max_vals - min_vals + 1e-8)  # (B,*)
        normalized_tensor = (
            normalized_tensor * (self.bounds_range[1] - self.bounds_range[0]) + self.bounds_range[0]
        )

        normalized_tensor[normalized_tensor < self.clips_min] = 0
        # Restore initial shape
        normalized_tensor = normalized_tensor.reshape(B, *extra_dims)

        return normalized_tensor  # (B,*)




class ComputeArbitraryFuncFromLenia(nn.Module):
    def __init__(self, mus: torch.Tensor, sigmas : torch.Tensor, n_coeffs : int, g : Callable, fft_points: int =1000, device:str='cpu'):
        super(ComputeArbitraryFuncFromLenia, self).__init__()
        self.g = g
        self.n_coeffs = n_coeffs
        self.mus = mus
        self.sigmas = sigmas
        self.u = torch.linspace(-1, 1, fft_points, device=device)
        self.period = self.get_period()
        self.shift = self.u.min()

    def get_period(self) -> torch.Tensor:
        return (self.u.max() - self.u.min())

    def set_mus(self, mus : torch.Tensor):
        self.mus = mus

    def set_sigmas(self, sigmas : torch.Tensor):
        self.sigmas = sigmas

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_values = self.g(self.u.expand(*self.mus.shape, -1), self.mus[..., None], self.sigmas[..., None])
        fft_coeffs = torch.fft.fft(f_values, dim=-1) / f_values.shape[-1]

        d = (self.u[1] - self.u[0]).item()
        freqs = torch.fft.fftfreq(f_values.shape[-1], d=d) * self.period
        indices = torch.argsort(freqs)
        mid_idx = f_values.shape[-1] // 2
        selected_indices = torch.cat([torch.arange(mid_idx - self.n_coeffs, mid_idx), torch.arange(mid_idx, mid_idx + self.n_coeffs + 1)])
        c_n = fft_coeffs[..., indices[selected_indices]]
        n_values = freqs[indices[selected_indices]]



        return c_n, n_values, self.period, self.shifts