import torch, torch.nn, torch.nn.functional as F
import numpy as np
from torchenhanced import DevModule
from .utils.noise_gen import perlin, perlin_fractal
from .utils.leniaparams import LeniaParams
import random, pygame
from .automaton import Automaton
from pathlib import Path
from .utils.funcgen import ArbitraryFunction


class MCLenia(DevModule, Automaton):
    """
    Batched Multi-channel lenia, to run batch_size worlds in parallel !
    Does not support live drawing in pygame, maybe will later.
    """

    def __init__(
        self,
        size,
        dt,
        num_channels=3,
        params=None,
        state_init=None,
        device="cpu",
        interest_files=None,
        save_dir=".",

    ):
        """
        Initializes automaton.

        Args :
            size : (B,H,W) of ints, size of the automaton and number of batches
            dt : time-step used when computing the evolution of the automaton
            num_channels : int, number of channels (C) in the automaton
            params : LeniaParams class, or dict of parameters containing the following
                keys-values :
                'k_size' : odd int, size of kernel used for computations
                'mu' : (B,C,C) tensor, mean of growth functions
                'sigma' : (B,C,C) tensor, standard deviation of the growth functions
                'beta' :  (B,C,C, # of rings) float, max of the kernel rings
                'mu_k' : (B,C,C, # of rings) [0,1.], location of the kernel rings
                'sigma_k' : (B,C,C, # of rings) float, standard deviation of the kernel rings
                'weights' : (B,C,C) float, weights for the growth weighted sum
            or :
                'k_size' : odd int, size of kernel used for computations
                'k_coeffs' : (B,C,C, # of harmonics) float, coefficients for the kernel harmonics
                'k_harmonics' : (B,C,C # of harmonics) float, values of harmonics
                'g_coeffs' : (B,C,C, # of harmonics) float, coeffs for growth harmonics
                'g_harmonics' : (B,C,C, # of harmonics) float, values of growth harmonics
            device : str, device
        """
        DevModule.__init__(self)
        Automaton.__init__(self, size[1:])
        self.to(device)

        self.batch = size[0]
        self.h, self.w = size[1:]
        self.C = num_channels

        if params is None:
            # Generates random parameters
            self.params = LeniaParams(batch_size=self.batch, k_size=25, channels=self.C, device=device)
        elif isinstance(params, dict):
            self.params = LeniaParams(param_dict=params, device=device)
        else:
            self.params = params

        self.k_size = self.params["k_size"]  # kernel size
        self.register_buffer("state", torch.rand((self.batch, self.C, self.h, self.w)))

        if state_init is None:
            self.set_init_fractal()  # Fractal perlin init
        else:
            self.state = state_init.to(self.device)  # Specific init

        self.dt = dt

        # Buffer for all parameters since we do not require_grad for them :
        self.register_buffer("mu", self.params["mu"])  # mean of the growth functions (B,C,C)
        self.register_buffer(
            "sigma", self.params["sigma"]
        )  # standard deviation of the growths functions (B,C,C)
        self.register_buffer("beta", self.params["beta"])  # max of the kernel rings (B,C,C, # of rings)
        self.register_buffer("mu_k", self.params["mu_k"])  # mean of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer(
            "sigma_k", self.params["sigma_k"]
        )  # standard deviation of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer(
            "weights", self.params["weights"]
        )  # raw weigths for the growth weighted sum (B,C,C)
        self.register_buffer("kernel", torch.zeros((self.k_size, self.k_size)))

        self.update_params(self.params)

        # For interactivity and visualization
        self.display_kernel = False
        self.save_dir = save_dir
        if interest_files is not None:
            self.interest_files = [file_path.as_posix() for file_path in Path(interest_files).rglob("*.pt")]
        else:
            self.interest_files = None
        self.chosen_interesting = 0

    def update_params(self, params: LeniaParams, k_size_override=None):
        """
        Updates parameters of the automaton.
        Changes batch size to match the one of provided params (take mu as reference)

        Args:
            params : LeniaParams
        """
        if isinstance(params, LeniaParams):
            params = params.param_dict

        self.mu = params.get("mu", self.mu)
        self.sigma = params.get("sigma", self.sigma)
        self.beta = params.get("beta", self.beta)
        self.mu_k = params.get("mu_k", self.mu_k)
        self.sigma_k = params.get("sigma_k", self.sigma_k)
        self.weights = params.get("weights", self.weights)
        self.k_size = params.get("k_size", self.k_size)  # kernel sizes (same for all)

        if k_size_override is not None:
            self.k_size = k_size_override

        if self.k_size % 2 == 0:
            self.k_size += 1
            print(f"Increased even kernel size to {self.k_size} to be odd")

        self.params = LeniaParams(param_dict=params, device=self.device)

        self.batch = self.mu.shape[0]  # update batch size
        self.k = self.compute_kernel()  # (B,C,C,k_size,k_size)
        self.growth = self.compute_growth()  # growth function, callable

        self.fft_kernel = self.kernel_to_fft(self.k)  # (B,C,C,h,w)

        # self.norm_weights() => not needed anymore, automatically normalized in LeniaParams

    def set_init_fractal(self):
        """
        Sets the initial state of the automaton using fractal perlin noise.
        Max wavelength is k_size*1.5, chosen a bit randomly
        """
        self.state = perlin_fractal(
            (self.batch, self.h, self.w),
            int(self.k_size * 1.5),
            device=self.device,
            black_prop=0.25,
            num_channels=self.C,
            persistence=0.4,
        )

    def set_init_perlin(self, wavelength=None):
        """
        Sets initial state using one-wavelength perlin noise.
        Default wavelength is 2*K_size
        """
        if not wavelength:
            wavelength = self.k_size
        self.state = perlin(
            (self.batch, self.h, self.w),
            [wavelength] * 2,
            device=self.device,
            num_channels=self.C,
            black_prop=0.25,
        )

    def set_init_circle(self, fractal=False, radius=None):
        if radius is None:
            radius = self.k_size * 3
        if fractal:
            self.state = perlin_fractal(
                (self.batch, self.h, self.w),
                int(self.k_size * 1.5),
                device=self.device,
                black_prop=0.25,
                num_channels=self.C,
                persistence=0.4,
            )
        else:
            self.state = perlin(
                (self.batch, self.h, self.w),
                [self.k_size] * 2,
                device=self.device,
                num_channels=self.C,
                black_prop=0.25,
            )
        X, Y = torch.meshgrid(
            torch.linspace(-self.h // 2, self.h // 2, self.h, device=self.device),
            torch.linspace(-self.w // 2, self.w // 2, self.w, device=self.device),
        )
        R = torch.sqrt(X**2 + Y**2)
        self.state = torch.where(R < radius, self.state, torch.zeros_like(self.state, device=self.device))

    def kernel_slice(self, r):
        """
        Given a distance matrix r, computes the kernel of the automaton.
        In other words, compute the kernel 'cross-section' since we always assume
        rotationally symmetric kernel

        Args :
        r : (k_size,k_size), value of the radius for each pixel of the kernel
        """
        # Expand radius to match expected kernel shape
        r = r[None, None, None, None]  # (1,1, 1, 1, k_size, k_size)
        r = r.expand(
            self.batch, self.C, self.C, self.mu_k.shape[3], -1, -1
        )  # (B,C,C,#of rings,k_size,k_size)

        mu_k = self.mu_k[..., None, None]  # (B,C,C,#of rings,1,1)
        sigma_k = self.sigma_k[..., None, None]  # (B,C,C,#of rings,1,1)

        K = torch.exp(-(((r - mu_k) / sigma_k) ** 2) / 2)  # (B,C,C,#of rings,k_size,k_size)

        beta = self.beta[..., None, None]  # (B,C,C,#of rings,1,1)
        K = torch.sum(beta * K, dim=3)  #

        return K  # (B,C,C,k_size, k_size)

    def compute_kernel(self, force_standard=False):
        """
        Computes the kernel given the current parameters. Uses in priority
        arbitrary function if provided, else uses the standard way.

        """
        xyrange = torch.linspace(-1, 1, self.k_size).to(self.device)

        X, Y = torch.meshgrid(
            xyrange, xyrange, indexing="xy"
        )  # (k_size,k_size),  axis directions is x increasing to the right, y increasing to the bottom
        r = torch.sqrt(X**2 + Y**2)  # (k_size,k_size)

        if "k_harmonics" in self.params.param_dict and not (force_standard):
            harmonics = self.params["k_harmonics"].reshape(
                self.batch * self.C * self.C, -1
            )  # (B*C*C,# of harmonics)
            coeffs = self.params["k_coeffs"].reshape(
                self.batch * self.C * self.C, -1
            )  # (B*C*C,# of harmonics)
            arbi = ArbitraryFunction(
                coefficients=coeffs, harmonics=harmonics, bounds_range=(0.0, 1.0), device=self.device
            )
            K = arbi(r[None].expand(self.batch * self.C * self.C, -1, -1))  # (B,C,C,k_size,k_size)
            K = K.reshape(self.batch, self.C, self.C, self.k_size, self.k_size)
            K = create_smooth_circular_mask(K, self.k_size // 2)
        else:
            K = self.kernel_slice(r)  # (B,C,C,k_size,k_size)

        # Normalize the kernel, s.t. integral(K) = 1
        summed = torch.sum(K, dim=(-1, -2), keepdim=True)  # (B,C,C,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed < 1e-6, 1, summed)
        K /= summed

        return K  # (B,C,C,k_size,k_size)

    def kernel_to_fft(self, K):
        # Pad kernel to match image size
        # For some reason, pad is left;right, top;bottom, (so W,H)
        K = F.pad(K, [0, (self.w - self.k_size)] + [0, (self.h - self.k_size)])  # (B,C,C,h,w)

        # Center the kernel on the top left corner for fft
        K = K.roll((-(self.k_size // 2), -(self.k_size // 2)), dims=(-1, -2))  # (B,C,C,h,w)

        K = torch.fft.fft2(K)  # (B,C,C,h,w)

        return K  # (B,C,C,h,w)

    def compute_growth(self, force_standard=False):
        """
        Constructs the growth function given current parameters.
        By default, uses the ArbitraryFunction way if the necessary parameters are defined
        """

        if "g_harmonics" in self.params.param_dict and not (force_standard):
            # Use ArbitraryFunction
            coeffs = self.params["g_coeffs"].reshape(
                self.batch * self.C * self.C, -1
            )  # (B*C*C,# of harmonics)
            harmonics = self.params["g_harmonics"].reshape(
                self.batch * self.C * self.C, -1
            )  # (B*C*C,# of harmonics)
            arbi = ArbitraryFunction(
                coefficients=coeffs, harmonics=harmonics, bounds_range=(-2.0, 2.0), device=self.device
            )

            def growth(u):
                B, C, C, H, W = u.shape
                u = u.reshape(B * C * C, H, W)
                out = arbi(u)
                return out.reshape(B, C, C, H, W)

            return growth
        else:
            mu = self.mu[..., None, None]  # (B,C,C,1,1)
            sigma = self.sigma[..., None, None]  # (B,C,C,1,1)
            mu = mu.expand(-1, -1, -1, self.h, self.w)  # (B,C,C,H,W)
            sigma = sigma.expand(-1, -1, -1, self.h, self.w)  # (B,C,C,H,W)
            growth = lambda u: 2 * torch.exp(-((u - mu) ** 2 / (sigma) ** 2) / 2) - 1
            return growth




    @torch.no_grad()
    def step(self):
        """
        Steps the automaton state by one iteration.
        """

        U = self.kernel_fftconv(self.state)  # (B,C,C,H,W)

        assert (self.h, self.w) == (U.shape[-2], U.shape[-1])

        weights = self.weights[..., None, None]  # (B,C,C,1,1)
        weights = weights.expand(-1, -1, -1, self.h, self.w)  # (B,C,C,H,W)

        # Weight normalized growth :
        dx = (self.growth(U) * weights).sum(
            dim=1
        )  # (B,C,H,W) # G(U)[:,i,j] is contribution of channel i to channel j

        # Apply growth and clamp
        self.state = torch.clamp(self.state + self.dt * dx, 0, 1)  # (B,C,H,W)

    def kernel_fftconv(self, state):
        """
        Compute convolution using fft_kernel
        """
        state = torch.fft.fft2(state)  # (B,C,H,W) fourier transform
        state = state[:, :, None]  # (B,1,C,H,W)
        state = state * self.fft_kernel  # (B,C,C,H,W), convoluted
        state = torch.fft.ifft2(state)  # (B,C,C,H,W), back to spatial domain

        return torch.real(state)

    def mass(self):
        """
        Computes average 'mass' of the automaton for each channel

        returns :
        mass : (B,C) tensor, mass of each channel
        """

        return self.state.mean(dim=(-1, -2))  # (B,C) mean mass for each color

    @torch.no_grad()
    def draw(self):
        """
        Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"

        toshow = self.state[0].clone()  # (C,H,W), pygame conversion done later

        if self.C == 1:
            toshow = toshow.expand(3, -1, -1)  # (3,H,W)
        elif self.C == 2:
            toshow = torch.cat([toshow, torch.zeros_like(toshow)], dim=0)  # (3,H,W)
        else:
            toshow = toshow[:3, :, :]  # (3,H,W)

        if self.display_kernel == True:
            kern = self.compute_ker()  # (C,3,k_size,k_size)
            for i in range(kern.shape[0]):
                toshow[:, self.h - self.k_size : self.h, i * self.k_size : (i + 1) * self.k_size] = kern[
                    i
                ].cpu()

        self._worldmap = torch.clamp(toshow, 0.0, 1.0)

    def process_event(self, event, camera=None):
        """
        N -> New random parameters
        A -> Random parameters using ArbitraryFunction
        M -> Load new interesting param
        U -> Variate around parameters
        I -> Intialize with fractal perlin
        J -> Initialize with perlin
        O -> Initialize with circle
        L -> Initialize with random wavelength perlin
        S -> Save the current parameters
        K -> Toggle display kernel
        DEL -> sets state to 0
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                """ New random parameters"""
                # params = param_gen(device)
                params = LeniaParams.random_gen(
                    batch_size=1, num_channels=self.C, device=self.device, k_size=31
                )
                # Probably should put the lines below in a function
                self.update_params(params, k_size_override=None)
            if event.key == pygame.K_a:
                params = LeniaParams.arbi_gen(
                    batch_size=1, num_channels=self.C, device=self.device, k_size=31
                )
                self.update_params(params, k_size_override=None)
            if event.key == pygame.K_u:
                """ Variate around parameters"""
                mutated_params = self.params.mutate(magnitude=0.1, rate=0.8)
                self.update_params(mutated_params, k_size_override=None)
            if event.key == pygame.K_i:
                # Intialize with fractal perlin
                self.set_init_fractal()
            if event.key == pygame.K_j:
                # Initialize with perlin
                self.set_init_perlin()
            if event.key == pygame.K_o:
                self.set_init_circle()
            if event.key == pygame.K_l:
                # Initialize with random wavelength perlin
                sq_size = random.randint(5, min(self.h, self.w))
                self.set_init_perlin(wavelength=sq_size)
            if event.key == pygame.K_m:
                if self.interest_files:
                    # Load random interesting param, if we have some
                    file = self.interest_files[self.chosen_interesting]  # To add as parameter
                    self.chosen_interesting = (self.chosen_interesting + 1) % len(self.interest_files)

                    params = LeniaParams(from_file=file, device=self.device)
                    self.update_params(params, k_size_override=None)
                    print("Loaded : ", file)
            if event.key == pygame.K_s:
                # Save the current parameters to remarkable dir :
                self.params.save_indiv(self.save_dir, annotation=["_nice"])
            if event.key == pygame.K_k:
                # Toggle display kernel
                self.display_kernel = not self.display_kernel
            if event.key == pygame.K_DELETE | pygame.K_BACKSPACE:
                self.state = torch.zeros_like(self.state)

    def compute_ker(self):
        """
        Prepares the kernel and translate it to an RGB image for viewing.

        returns :
        kern : (C,3,k_size,k_size) tensor, kernel as
        """
        kern = self.k[0].detach()  # (C,C, k_size, k_size), removed batch

        if kern.shape[1] == 1:
            kern = kern.expand(1, 3, -1, -1)
        elif kern.shape[1] > 3:
            kern = kern[:3, :3]  # Cut, and include only the first 3 set of kernels

        maxs = torch.tensor([torch.max(kern[i]) for i in range(kern.shape[0])], device=self.device)  # (C,)
        # print(maxs)
        maxs = maxs[:, None, None, None]
        kern = kern / maxs

        return kern  # (C,3,k_size,k_size)


def create_smooth_circular_mask(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    H, W = tensor.shape[-2], tensor.shape[-1]
    center_y = (H - 1) / 2  # Allow fractional center for better smoothness
    center_x = (W - 1) / 2
    y = torch.linspace(0, H - 1, H, device=tensor.device).view(-1, 1)
    x = torch.linspace(0, W - 1, W, device=tensor.device).view(1, -1)
    distance = ((y - center_y) ** 2 + (x - center_x) ** 2).sqrt()
    smooth_transition = 0.5  # Define a region for the smooth transition (around the edge of the circle)
    mask = torch.clamp(1 - (distance - radius) / smooth_transition, 0, 1)
    masked_tensor = tensor * mask

    return masked_tensor
