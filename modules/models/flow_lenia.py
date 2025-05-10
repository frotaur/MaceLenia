import torch, torch.nn, torch.nn.functional as F
import pygame
from .lenia import Lenia
import random


def sobel_x(x):
    k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda:0").tile(
        (x.shape[1], 1, 1, 1)
    )

    sx = torch.nn.functional.conv2d(x, k_x, groups=x.shape[1], stride=1, padding="same")

    return sx.permute((0, 2, 3, 1))


def sobel_y(x):
    k_y = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda:0").T.tile(
        (x.shape[1], 1, 1, 1)
    )
    sy = torch.nn.functional.conv2d(x, k_y, groups=x.shape[1], stride=1, padding="same")

    return sy.permute((0, 2, 3, 1))


def sobel(x):
    sx = sobel_x(x.to(torch.float32))

    sy = sobel_y(x.to(torch.float32))
    sxy = torch.cat((sy[:, :, :, None, :], sx[:, :, :, None, :]), dim=3)
    return sxy


def construct_mesh_grid(X, Y):
    x, y = torch.arange(X), torch.arange(Y)
    mx, my = torch.meshgrid(x, y)
    pos = torch.dstack((mx, my)) + 0.5
    # pos = pos.permute((2,1,0))
    return pos.to("cuda:0")


def construct_ds(dd):
    dxs = []
    dys = []
    for dx in range(-dd, dd + 1):
        for dy in range(-dd, dd + 1):
            dxs.append(dx)
            dys.append(dy)
    dxs = torch.tensor(dxs, device="cuda:0")
    dys = torch.tensor(dys, device="cuda:0")
    return dxs, dys


class ReintegrationTracker:
    def __init__(self, X, Y, dt, dd=2, sigma=0.65):
        self.X = X
        self.Y = Y
        self.dd = dd
        self.dt = dt
        self.sigma = sigma
        self.pos = construct_mesh_grid(X, Y)[None, ...]
        self.dxs, self.dys = construct_ds(dd)

    def step(self, grid, mu, dx, dy):
        gridR = torch.roll(grid.permute(0, 2, 3, 1), (dx, dy), (1, 2))

        muR = torch.roll(mu, (dx, dy), (1, 2))

        dpmu = (self.pos[..., None] - muR).abs()

        sz = 0.5 - dpmu + self.sigma

        area = torch.prod(torch.clip(sz, 0, min(1.0, 2 * self.sigma)), dim=-2) / (4 * self.sigma**2)

        ngrid = gridR * area

        return ngrid.permute(0, 3, 1, 2)

    def apply(self, grid, F):
        ma = self.dd - self.sigma

        mu = self.pos[..., None] + (self.dt * F).clip(-ma, ma)

        mu = torch.clip(mu, self.sigma, self.X - self.sigma)
        ngrid = torch.stack([self.step(grid, mu, dx, dy) for dx, dy in zip(self.dxs, self.dys)])

        return ngrid.sum(dim=0)


class FlowLenia(Lenia):
    """Pytorch port of mass conserving FlowLenia"""

    def __init__(
        self,
        size,
        dt,
        num_channels=3,
        params=None,
        state_init=None,
        device="cpu",
        dd=2,
        sigma_rt=0.65,
        has_food=False,
        interest_files=None,
        save_dir=".",
    ):
        self.dd = dd
        self.sigma_rt = sigma_rt
        self.theta_x = 2
        self.n = 2
        self.rt = ReintegrationTracker(size[1], size[2], dt, dd=self.dd, sigma=self.sigma_rt)
        self.has_food = has_food
        super().__init__(
            size,
            dt,
            num_channels,
            params,
            state_init,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir,
        )

        self._temp = 1
        self.Aff = self.compute_affinity()
        self.params.dd = self.dd
        self.params.sigma_rt = self.sigma
        self.params.theta_x = self.theta_x

    def compute_affinity(self):
        Aff = self.kernel_fftconv(self.state)  # (B,C,C,H,W) first step affinity, usual convolutions
        weights = self.weights[..., None, None]  # (B,C,C,1,1)
        Aff = (self.growth(Aff) * weights).sum(dim=1)  # (B,C,H,W) pre-exponential affinity

        grad_u = sobel(Aff)  # (B,C,2,H,W)

        grad_x = sobel(self.state.sum(dim=1, keepdims=True))

        # added a sum over the channel in the alpha computation, as in the paper
        alpha = (
            (self.state.permute(0, 2, 3, 1)[:, :, :, None, :].sum(dim=-1, keepdims=True) / self.theta_x)
            ** self.n
        ).clip(0, 1)

        F = grad_u * (1 - alpha) - grad_x * alpha
        # F= grad_u

        return F

    def update_food(self):
        """uncomment the death sections for death mechanics, but its finicky and i dont like it"""
        where_food = self.food_channel > 0  # Where the food channels are
        where_contact = (
            self.state.sum(dim=1)[:, None, :, :] > 0.1
        )  # Where the eating channel is, we could amke this dynamic, 0.1 is the threshold for eating
        death = (
            (self.state.sum(dim=1)[:, None, :, :] < 0.01) & (self.state.sum(dim=1)[:, None, :, :] > 0)
        ) * self.state  # death of the feeding channel, very finicky

        overlap = where_food & where_contact  # where the channels overlap
        transfer = (
            torch.ones_like(where_food) * overlap * 0.03
        )  # How much to increase / deacrease the mass currently set to 0.01
        self.state += transfer  # Lenia mass increase
        self.state -= death
        self.food_channel -= transfer  # Food mass deacrease
        self.food_channel += death.sum(dim=1)[:, None, :, :] / 3

    @torch.no_grad()
    def step(self):
        Aff = self.compute_affinity()

        self.state = self.rt.apply(self.state, Aff)
        if self.has_food:
            self.update_food()

    def temp(self):
        return self._temp

    def process_event(self, event, camera=None):
        """
        UP -> Increase temperature
        DOWN -> Decrease temperature
        RIGHT-> Increase sigma
        LEFT -> Decrease sigma
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.theta_x += 0.2
            if event.key == pygame.K_DOWN:
                self.theta_x -= 0.2

            if event.key == pygame.K_RIGHT:
                self.sigma_rt += 0.02
                self.rt.sigma = self.sigma_rt
            if event.key == pygame.K_LEFT:
                self.sigma_rt -= 0.02
                self.rt.sigma = self.sigma_rt

    process_event.__doc__ = Lenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )  # Hack to append the docstring of MCLenia.process_event

    def get_string_state(self):
        return (
            f"total mass: {self.state.sum().item():.2f}, temp: {self.theta_x:.2f}, sigma: {self.sigma_rt:.2f}"
        )

    def random_food_chan(self, num_spots=100, food_size=5):
        """
        Returns a food channel with num_spots of food of size food_size
        Args :
            num_spots : int, number of food spots
            food_size : int, size of the food spots

        Returns :
            food_chan : tensor, (B,1,H,W) food channel
        """
        places = [
            [random.randint(food_size, self.h - food_size), random.randint(food_size, self.w - food_size)]
            for _ in range(num_spots)
        ]
        food_chan = torch.zeros((self.batch, 1, self.h, self.w), device=self.device)
        for place in places:
            food_chan[
                :, :, place[0] - food_size : place[0] + food_size, place[1] - food_size : place[1] + food_size
            ] = 1

        return food_chan

    def set_init_fractal(self):
        super().set_init_fractal()
        if self.has_food:
            self.food_channel = self.random_food_chan()  # (B,1, H,W)

    def set_init_perlin(self, wavelength=None):
        super().set_init_perlin(wavelength)
        if self.has_food:
            self.food_channel = self.random_food_chan()  # (B,1, H,W)

    def set_init_circle(self, fractal=False, radius=None):
        super().set_init_circle(fractal, radius)
        if self.has_food:
            self.food_channel = self.random_food_chan()  # (B,1, H,W)

    @torch.no_grad()
    def draw(self):
        """
        Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"

        toshow = self.state[0].clone()  # (C,H,W), pygame conversion done later

        if self.C == 1:
            toshow = toshow.repeat(3, 1, 1)  # (3,H,W)
        elif self.C == 2:
            toshow = torch.cat([toshow, torch.zeros_like(toshow)], dim=0)  # (3,H,W)
        else:
            toshow = toshow[:3, :, :]  # (3,H,W)

        if self.has_food:
            toshow[:, :, :] += self.food_channel[0]  # (1,H,W)

        if self.display_kernel == True:
            kern = self.compute_ker()  # (C,3,k_size,k_size)
            for i in range(kern.shape[0]):
                toshow[:, self.h - self.k_size : self.h, i * self.k_size : (i + 1) * self.k_size] = kern[
                    i
                ].cpu()

        self._worldmap = torch.clamp(toshow, 0.0, 1.0)
