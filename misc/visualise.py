import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda:0"
def perchannel_conv(x, filters):
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)

ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
ones = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=DEVICE)
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device=DEVICE)
lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device=DEVICE)
gaus = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device=DEVICE)

def perception(x):
    filters = torch.stack([sobel_x, sobel_x.T, lap])
    obs = perchannel_conv(x, filters)
    return torch.cat((x,obs), dim = 1 )


class MassConservingNCA(torch.nn.Module):
    def __init__(self,C,hidden_n, device):
        super(MassConservingNCA, self).__init__()
        self.C = C
        self.w1 = torch.nn.Conv2d(4 * C, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, C, 1, bias=False)
        self.w2.weight.data.zero_()
        self.device = device

    def forward(self, x, update_rate=1):
        y = perception(x)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=self.device) + update_rate).floor()
        x_normal = x[:,3:,...]
        x_mass = x[:,:3,...]

        y_normal = y[:,3:,...]
        y_mass = y[:,:3,...]

        Aff = torch.exp(y_mass*0.1)

        x_mass = self.redistribution(Aff, x_mass)
        x_normal = x_normal + y_normal * update_mask

        x = torch.cat((x_mass, x_normal), dim = 1)

        return x

    def redistribution(self,Aff,state):

        B, C, H, W = state.shape
        Aff_exp = F.pad(Aff, (1, 1, 1, 1), mode="circular")  # (B,C,H+2,W+2) for the (3,3) kernel
        Aff_exp = F.unfold(Aff_exp, kernel_size=(3, 3)).reshape(B, C, 9, H, W)  # (B,C*9,H,W)
        E = Aff_exp.sum(dim=2)
        E_exp = F.pad(E, (1, 1, 1, 1), mode="circular")
        E_exp = F.unfold(E_exp, kernel_size=(3, 3)).reshape(B, C, 9, H, W)  # (B,C*9,H,W)
        state_exp = F.pad(state, (1, 1, 1, 1), mode="circular")
        state_exp = F.unfold(state_exp, kernel_size=(3, 3)).reshape(B, C, 9, H, W)  # (B,C*9,H,W)

        state = ((Aff[:, :, None, ...] / E_exp) * state_exp).sum(dim=2)

        min_aff = Aff.min()
        max_aff = Aff.max()
        Aff_norm = (Aff - min_aff) / (max_aff - min_aff+1e-6)
        Aff_c = Aff_norm / (Aff_norm.sum(dim=1, keepdim=True)+1e-6)



        target_cross_c_masses = state.sum(dim=1, keepdim=True) * Aff_c
        diff_target = state - target_cross_c_masses
        alpha = 0.0001
        state -= diff_target * alpha

        return state

#%%
def show_batch(results, channels=4):
    x = results.cpu().clone().permute((0, 2, 3, 1)).detach().numpy()
    plt.figure(2)
    plt.clf()
    num = results.shape[0]
    if num > 8:
        num = 8
    for i in range(num):
        img = x[i, :, :, 0:channels]
        plt.figure(2)
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)


def get_batch(pool, x_prime, batch_size):
    idxs = np.random.randint(0, pool.shape[0], batch_size)
    batch = pool[idxs, :, :, :]
    batch[0:2, :, :, :] = x_prime
    return batch, idxs


def update_pool(pool, results, idxs):
    pool[idxs] = results.clone().detach()
    return pool


def get_image(path,height=50, width=50, padding =0):
    base = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    base = cv2.resize(base, (int(height), int(width)), interpolation=cv2.INTER_LINEAR)
    base_2 = base / 255
    base_2[..., :3] *= base_2[..., 3:]
    base_torch = torch.tensor(base_2, dtype=torch.float32, requires_grad=True).permute((2, 0, 1)).to(DEVICE)
    base_torch = torch.nn.functional.pad(base_torch, [padding,padding,padding,padding ])
    base_tt = base_torch.cpu().permute((1, 2, 0)).clone().detach().numpy()
    return base_torch,base_tt

def get_reference_image_and_seed(path, height = 50, width =50, channels =16):
    base = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    base = cv2.resize(base, (int(height), int(width)), interpolation=cv2.INTER_LINEAR)
    base_2 = base / 255
    base_2[..., :3] *= base_2[..., 3:]
    base_torch = torch.tensor(base_2, dtype=torch.float32, requires_grad=True).permute((2, 0, 1)).to(DEVICE)
    x_prime = torch.zeros((channels, height, width), dtype=torch.float32).to(DEVICE)
    x_prime[:3, int(height / 2), int(width / 2)] = 233
    return base_torch, x_prime

def to_vue_image(tensor):
    return tensor.cpu().permute((1, 2, 0)).clone().detach().numpy()


def double_mass(state, max):
    B, C, H, W = state.shape
    n_state = [state[i,:3].clone() *2 if (state[i,:3].sum() *2) < max else state[i,:3] for i in range(B)]
    n_state = torch.stack(n_state, dim=0)

    return torch.cat((n_state, state[:,3:]), dim=1)

#%%

HEIGHT = 50
WIDTH = 50
CHANNELS = 16 #<--- NCA feature channels
BATCH_SIZE = 1


nca = MassConservingNCA(C=CHANNELS, hidden_n=96, device=DEVICE)
nca.load_state_dict(torch.load("Growing.pth"))
nca.to(DEVICE).eval()



seed = torch.zeros((BATCH_SIZE, CHANNELS ,HEIGHT, WIDTH), device=DEVICE)
seed[:,:3,int(HEIGHT/ 2), int(WIDTH / 2)] = 100
x = seed
for i in range(60000):

    if ((i % 10) == 0) and (i > 0):
        with torch.no_grad():
            x = double_mass(x, 701)

    x = nca(x)

    x = x.detach()
    image = x.clone().detach().cpu().permute(0, 3, 2, 1).numpy()[0, :, :, :4]
    image = cv2.resize(image, [500, 500], interpolation=cv2.INTER_LINEAR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    cv2.imshow("image", image)

    cv2.waitKey(1)
    # print(i)