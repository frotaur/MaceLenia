from collections.abc import Callable
import torch
import matplotlib.pyplot as plt

from torch import nn




class ComputeArbitraryFuncFromLenia(nn.Module):
    def __init__(self, mus: torch.Tensor, sigmas : torch.Tensor, n_coeffs : int, g : Callable, fft_points: int =1000, device:str='cpu'):
        super(ComputeArbitraryFuncFromLenia, self).__init__()
        self.g = g
        self.n_coeffs = n_coeffs
        self.mus = mus
        self.sigmas = sigmas
        self.u = torch.linspace(-1, 1, fft_points, device=device)
        self.period = self.get_period()
        self.shifts = self.u.min()

    def get_period(self) -> torch.Tensor:
        return (self.u.max() - self.u.min())

    def set_mus(self, mus : torch.Tensor):
        self.mus = mus

    def set_sigmas(self, sigmas : torch.Tensor):
        self.sigmas = sigmas

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_values = self.g(u_1.expand(*self.mus.shape, -1), self.mus[..., None], self.sigmas[..., None])
        fft_coeffs = torch.fft.fft(f_values, dim=-1) / f_values.shape[-1]

        d = (self.u[1] - self.u[0]).item()
        freqs = torch.fft.fftfreq(f_values.shape[-1], d=d) * self.period
        indices = torch.argsort(freqs)
        mid_idx = f_values.shape[-1] // 2
        selected_indices = torch.cat([torch.arange(mid_idx - self.n_coeffs, mid_idx), torch.arange(mid_idx, mid_idx + self.n_coeffs + 1)])
        c_n = fft_coeffs[..., indices[selected_indices]]
        n_values = freqs[indices[selected_indices]]


        return c_n, n_values, self.period, self.shifts


def compute_fourier_coefficients_fft(f, u,mu,sigma, period, N):   # returns coeficinets (B,*,Ncoefs) Ncoef = N*2 +1, and those frequency values
    #N = N*2 +1

    f_values = f(u_1.expand(*mu.shape, -1),mu[...,None],sigma[...,None])
    fft_coeffs = torch.fft.fft(f_values, dim=-1) / f_values.shape[-1]
    fft_magnitude = torch.abs(fft_coeffs)
    d = (u[1] - u[0]).item()
    freqs = torch.fft.fftfreq(f_values.shape[-1], d=d) * period
    indices = torch.argsort(freqs)
    mid_idx = f_values.shape[-1] // 2
    selected_indices = torch.cat([torch.arange(mid_idx - N, mid_idx), torch.arange(mid_idx, mid_idx + N + 1)])
    #mag_sorted_indices = torch.argsort(fft_magnitude, dim=-1, descending=True)

    # Select the top N coefficients based on the sorted magnitude
    #selected_indices = mag_sorted_indices[..., :N]
    print(selected_indices.shape)
    c_n = fft_coeffs[ ...,indices[selected_indices]]
    n_values = freqs[indices[selected_indices]]
    print(freqs.shape)
    #c_n = torch.gather(fft_coeffs, dim=-1, index=selected_indices)
    #n_values = torch.gather(freqs.tile(selected_indices.shape[0],1), dim=-1, index=selected_indices)
    return c_n, n_values


def reconstruct_from_fourier(c_n, n_values, u, period, shifts):

    #shift = u.min().unsqueeze(-1)  # Get first value of `u`, keep batch dims
    c_n = c_n.unsqueeze(-1)
    n_values = n_values.unsqueeze(-1)
    print(c_n.shape)
    print(u.shape)
    reconstruction = torch.sum(
        c_n * torch.exp(1j * 2 * n_values * torch.pi * (u - shifts) / period),
        dim=-2
    )


    return reconstruction.real  # Take real part



growth = lambda u, mu, sigma: 2 * torch.exp(-((u - mu) ** 2 / (sigma) ** 2) / 2) - 1
C = 3
B = 4

mu_1 = torch.rand((B*C*C))*0.7
sigma_1 = torch.rand((B*C*C))*0.05
N_coeffs = 22

computer = ComputeArbitraryFuncFromLenia(mu_1, sigma_1, N_coeffs, growth)
u_1 = torch.linspace(-1, 1, 1000)
#period = (u_1.max() - u_1.min()).item()
fourier_coeffs, n_values, period, shifts  = computer()
recon = reconstruct_from_fourier(fourier_coeffs, n_values, u_1, period, shifts)  #





res = growth(u_1.expand(*mu_1.shape, -1), mu_1[...,None], sigma_1[...,None])
# Plot first batch sample
plt.plot(u_1, res[2], label="Original Function", linestyle="dashed")
plt.plot(u_1, recon[2], color='red', label="Reconstructed")
plt.plot(u_1, res[1], label="Original Function", linestyle="dashed")
plt.plot(u_1, recon[1], color='red', label="Reconstructed")
plt.legend()
plt.xlabel("u")
plt.ylabel("Function Value")
plt.title("Fourier Series Reconstruction with Batch Support")
plt.show()