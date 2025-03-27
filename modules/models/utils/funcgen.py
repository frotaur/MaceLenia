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
        ranges: Tensor,
        rescale: tuple = None,
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
            ranges: (func_num,2) tensor of function x ranges. E.g., if a funciton is defined in [-1,1], the range is (-1,1)
            rescale: tuple, (min,max). Will rescale the function values to this range
            clips_min: float, after rescaling will clip to this value
            device : torch device
        """
        super().__init__()
        self.device = device
        assert coefficients.shape == harmonics.shape, "Coefficients and harmonics must have the same shape"
        self.func_num = coefficients.shape[0]

        self.rescale_range = rescale
        self.clips_min = clips_min
        ranges = ranges.to(self.device)
        self.shifts = ranges[:, 0] # (B,)
        self.period = ranges[:, 1] - ranges[:, 0] # (B,)


        # Set the coefficients and harmonics as buffers
        # Change the following to nn.Parameter if we want to train the coefficients
        self.register_buffer("coefficients", coefficients.to(self.device))  # (B,num_harmonics)
        self.register_buffer("harmonics", harmonics.to(self.device))  # (B,num_harmonics)

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
            * torch.exp(1j * 2 * torch.pi * self.harmonics[..., None] * (x - self.shifts[:,None,None]) / self.period[:,None,None]),
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
            values[values < self.clips_min] = 0.

        # Restore initial shape
        values = values.reshape(B, *extra_dims)

        return values  # (B,*)

    @staticmethod
    def random_arbi(func_num, n_coeffs, ranges, rescale=None, clips_min=None, device="cpu"):
        """
        Returns random arbitrary function, samples coefficients from a Gaussian distribution

        Args:
            func_num : int, number of functions to generate
            n_coeffs : int, will use n_coeffs*2+1 harmonics
            ranges : tuple or (func_num,2) tensor of function x ranges. E.g., if a funciton is defined in [-1,1], the range is (-1,1)
            rescale : tuple, (min,max) range of the x values for the function to be rescaled to
            clips_min : float, after rescaling will clip to this value
            device : torch device
        """
        num_harmonics = 2 * n_coeffs + 1  # (num_harmonics,)
        # Randomly create harmonics and coeffs. Makes sure the coefficients are hermitian s.t. the function is real
        coefficients = torch.randn(
            func_num, n_coeffs, device=device
        )+1j*torch.randn(
            func_num, n_coeffs, device=device
        )


        coefficients = torch.cat([torch.randn(func_num,1, device=device),coefficients, torch.conj(coefficients.flip(dims=(1,)))], dim=1) # (func_num,num_harmonics)
        
        for i in range(coefficients.shape[1]):
            assert (coefficients[:,i]==torch.conj(coefficients[:,-i])).all()

        harmonics = torch.fft.fftfreq(num_harmonics,d=1/num_harmonics) # (num_harmonics,) integer frequencies
        harmonics = harmonics[None].expand(func_num, num_harmonics) # (func_num,num_harmonics)

        # harmonics = torch.tensor(
        #     range(-(num_harmonics - 1) // 2, (num_harmonics - 1) // 2 + 1), dtype=float, device=device
        # )[None].expand(func_num, num_harmonics)

        if(isinstance(ranges, tuple)):
            ranges = torch.tensor([ranges[0], ranges[1]], device=device)[None,:].expand(func_num, -1)

        return ArbitraryFunction(
            coefficients=coefficients,
            harmonics=harmonics,
            ranges=ranges,
            rescale=rescale,
            clips_min=clips_min,
            device=device
        )

    @staticmethod
    def from_function_evals(func_evals: torch.Tensor, x_bounds: tuple[float], n_coeffs: int, device='cpu'):
        """
        Given a (batched) tensor of function evaluations, an
        ArbitraryFunction instance approximating the function 
        evaluations with 2*n_coeffs+1 harmonics.

        Args:
            func_evals : (N,T), tensor of 1D function evaluations. NOTE: 
                This should be the function evaluated at values x_t = t/T*(x_max-x_min) for t=0,1,...,T-1
            x_bounds : tuple, (min,max) range of the x values for the function
            n_coeffs : int, number of coefficients to use for the approximation
        """
        N, T = func_evals.shape

        func_evals = func_evals.to(device)  # We don't need to compute this on the GPU
        ranges = torch.tensor([x_bounds[0], x_bounds[1]], device=device)[None,:].expand(N, -1) # (N,2)

        fft_coeffs = torch.fft.fft(func_evals, dim=-1) / T # We already rescale the coeffs so they work with the inverse
        freqs = torch.fft.fftfreq(T,d=1/T)  # The harmonic frequencies, un-normalized (we scale by period later)

        assert 2 * n_coeffs + 1 <= T, "The number of coefficients is too high for the number of samples"
        main_freqs = torch.cat(
            [freqs[: n_coeffs + 1], freqs[-n_coeffs:]]
        )  # (2*n_coeffs+1,)The main frequencies we want to keep

        main_freqs = main_freqs[None].expand(N, -1)  # (N,2*n_coeffs+1)

        mains_coeffs = torch.cat(
            [fft_coeffs[:, : n_coeffs + 1], fft_coeffs[:, -n_coeffs:]], dim=-1
        )  # (N,2*n_coeffs+1)

        return ArbitraryFunction(
            coefficients= mains_coeffs,
            harmonics= main_freqs,
            ranges= ranges,
            rescale= None,
            clips_min= None,
            device= device,
        )

    @staticmethod
    def from_function(function:callable,x_bounds:tuple[float], n_coeffs:int, n_points:int=1000,device='cpu'):
        """
            Given a callable function that accepts a tensor of shape (T,) of evaluations, computes
            the arbitrary function that approximates the function with 2*n_coeffs+1 harmonics.
        """
        x = torch.arange(n_points,device=device).float() / n_points * (x_bounds[1]-x_bounds[0]) + x_bounds[0]# (T,)
        func_evals = function(x) # (T)

        return ArbitraryFunction.from_function_evals(func_evals[None],x_bounds,n_coeffs,device)

# class ComputeArbitraryFuncFromLenia(nn.Module):
#     def __init__(
#         self,
#         mus: torch.Tensor,
#         sigmas: torch.Tensor,
#         n_coeffs: int,
#         g: Callable,
#         fft_points: int = 1000,
#         device: str = "cpu",
#     ):
#         super(ComputeArbitraryFuncFromLenia, self).__init__()
#         self.g = g
#         self.n_coeffs = n_coeffs
#         self.mus = mus
#         self.sigmas = sigmas
#         self.u = torch.linspace(-1, 1, fft_points, device=device)
#         self.period = self.get_period()
#         self.shift = self.u.min()

#     def get_period(self) -> torch.Tensor:
#         return self.u.max() - self.u.min()

#     def set_mus(self, mus: torch.Tensor):
#         self.mus = mus

#     def set_sigmas(self, sigmas: torch.Tensor):
#         self.sigmas = sigmas

#     def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         f_values = self.g(self.u.expand(*self.mus.shape, -1), self.mus[..., None], self.sigmas[..., None])
#         fft_coeffs = torch.fft.fft(f_values, dim=-1) / f_values.shape[-1]

#         d = (self.u[1] - self.u[0]).item()
#         freqs = torch.fft.fftfreq(f_values.shape[-1], d=d) * self.period
#         indices = torch.argsort(
#             freqs
#         )  # Instead of sorting, we can take the first n_coeffs, then the n_coeffs starting from the end, it's what we want
#         mid_idx = f_values.shape[-1] // 2
#         selected_indices = torch.cat(
#             [
#                 torch.arange(mid_idx - self.n_coeffs, mid_idx),
#                 torch.arange(mid_idx, mid_idx + self.n_coeffs + 1),
#             ]
#         )
#         c_n = fft_coeffs[..., indices[selected_indices]]
#         n_values = freqs[indices[selected_indices]]

#         return c_n, n_values, self.period, self.shifts
