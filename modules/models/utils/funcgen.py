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
        coefficients: Tensor,
        harmonics: Tensor,
        period: Tensor,
        shifts: Tensor = torch.tensor([0]),
        rescale_range: tuple = None,
        clips_min=None,
        device="cpu",
    ):
        """
        If coefficient and harmonics are not provided, they are randomly generated. Both of them
        must be provided if one is, otherwise it will raise an error. They will override the num_harmonics
        and func_num values if provided.
        Args:
            coefficients: optional, (func_num, num_harmonics) tensor of coefficients to use
            harmonics: optional, (func_num, num_harmonics) tensor of harmonics to use
            period: (func_num,) tensor of function periods (i.e., the range of the function)
            shifts: default function range is [0,period]. Provide shift to change it to [shift, shift+period]
            rescale_range: tuple, (min,max). Will rescale the function values to this range
            clips_min: float, after rescaling will clip to this value
            device : torch device
        """
        super().__init__()
        self.device = device
        assert coefficients.shape == harmonics.shape, "Coefficients and harmonics must have the same shape"
        self.func_num = coefficients.shape[0]

        self.rescale_range = rescale_range
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

        values = torch.sum(
            self.coefficients[..., None]
            * torch.exp(1j * 2 * torch.pi * self.harmonics[..., None] * (x - self.shifts) / self.period),
            dim=1,
        ).real

        if(self.rescale_range is not None):
            min_vals, _ = values.min(dim=-1, keepdim=True)  # (B,1)
            max_vals, _ = values.max(dim=-1, keepdim=True)  # (B,1)

            values = (values - min_vals) / (max_vals - min_vals + 1e-8)  # (B,*)
            values = (
                values * (self.rescale_range[1] - self.rescale_range[0]) + self.rescale_range[0]
            )
        if(self.clips_min is not None):
            values[values < self.clips_min] = 0

        # Restore initial shape
        values = values.reshape(B, *extra_dims)

        return values  # (B,*)

    @staticmethod
    def random_arbi(func_num, num_harmonics, bounds_range=None, clips_min=None, device="cpu"):
        """
        Returns random arbitrary function, samples coefficients from a Gaussian distribution
        """
        # Randomly create harmonics and coeffs. Maybe in the future, add more parameters to sample harmonics ?
        coefficients = torch.randn(
            func_num, num_harmonics, device=device
        )  # Random, possibly negative, coefficients
        harmonics = torch.tensor(
            range(-(num_harmonics - 1) // 2, (num_harmonics - 1) // 2 + 1), dtype=float, device=device
        )[None].expand(func_num, num_harmonics)

        return ArbitraryFunction(
            func_num=func_num,
            num_harmonics=num_harmonics,
            coefficients=coefficients,
            harmonics=harmonics,
            bounds_range=bounds_range,
            clips_min=clips_min,
            device=device,
        )

    @staticmethod
    def from_function(func_evals: torch.Tensor, x_bounds: tuple[float], n_coeffs: int, device='cpu'):
        """
        Given a (batched) tensor of function evaluations, an
        ArbitraryFunction instance approximating the function 
        evaluations with 2*n_coeffs+1 harmonics.

        Args:
            func_evals : (N,T), tensor of 1D function evaluations
            x_bounds : tuple, (min,max) range of the x values for the function
            n_coeffs : int, number of coefficients to use for the approximation
        """
        N, T = func_evals.shape

        func_evals = func_evals.to("cpu")  # We don't need to compute this on the GPU
        period = x_bounds[1] - x_bounds[0]
        shifts = torch.tensor([x_bounds[0]]).expand(N)

        fft_coeffs = torch.fft.fft(func_evals, dim=-1) / T
        freqs = torch.fft.fftfreq(T)  # The harmonic frequencies

        assert 2 * n_coeffs + 1 <= T, "The number of coefficients is too high for the number of samples"
        main_freqs = torch.cat(
            [freqs[: n_coeffs + 1], freqs[-n_coeffs:]]
        )  # (2*n_coeffs+1,)The main frequencies we want to keep
        main_freqs = main_freqs[None].expand(N, -1)  # (N,2*n_coeffs+1)
        mains_coeffs = torch.cat(
            [fft_coeffs[:, : n_coeffs + 1], fft_coeffs[:, -n_coeffs:]], dim=-1
        )  # (N,2*n_coeffs+1)

        return ArbitraryFunction(
            func_num= N,
            num_harmonics= 2 * n_coeffs + 1,
            coefficients= mains_coeffs,
            harmonics= main_freqs,
            shifts= shifts,
            period= period,
            bounds_range= None,
            clips_min= None,
            device= device,
        )


class ComputeArbitraryFuncFromLenia(nn.Module):
    def __init__(
        self,
        mus: torch.Tensor,
        sigmas: torch.Tensor,
        n_coeffs: int,
        g: Callable,
        fft_points: int = 1000,
        device: str = "cpu",
    ):
        super(ComputeArbitraryFuncFromLenia, self).__init__()
        self.g = g
        self.n_coeffs = n_coeffs
        self.mus = mus
        self.sigmas = sigmas
        self.u = torch.linspace(-1, 1, fft_points, device=device)
        self.period = self.get_period()
        self.shift = self.u.min()

    def get_period(self) -> torch.Tensor:
        return self.u.max() - self.u.min()

    def set_mus(self, mus: torch.Tensor):
        self.mus = mus

    def set_sigmas(self, sigmas: torch.Tensor):
        self.sigmas = sigmas

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_values = self.g(self.u.expand(*self.mus.shape, -1), self.mus[..., None], self.sigmas[..., None])
        fft_coeffs = torch.fft.fft(f_values, dim=-1) / f_values.shape[-1]

        d = (self.u[1] - self.u[0]).item()
        freqs = torch.fft.fftfreq(f_values.shape[-1], d=d) * self.period
        indices = torch.argsort(
            freqs
        )  # Instead of sorting, we can take the first n_coeffs, then the n_coeffs starting from the end, it's what we want
        mid_idx = f_values.shape[-1] // 2
        selected_indices = torch.cat(
            [
                torch.arange(mid_idx - self.n_coeffs, mid_idx),
                torch.arange(mid_idx, mid_idx + self.n_coeffs + 1),
            ]
        )
        c_n = fft_coeffs[..., indices[selected_indices]]
        n_values = freqs[indices[selected_indices]]

        return c_n, n_values, self.period, self.shifts
