from modules import DiffusionLenia, DiffusionLeniaCrossChannel
from modules.models.utils import LeniaParams
from tqdm import tqdm
import torch
model = DiffusionLeniaCrossChannel(size=(1,800,800), dt=0.1, num_channels=3, device='cuda', has_food=True, save_dir='intrinsic')

def new_params(model, default=True):
    if(not default):
        params = LeniaParams.random_gen(
            batch_size=model.batch, num_channels=model.C, device=model.device, k_size=model.k_size,
            k_mult=model.k_mult
        )
    else:
        params = LeniaParams.default_gen(
            batch_size=model.batch, num_channels=model.C, device=model.device, k_size=model.k_size,
            k_mult=model.k_mult
        )
    model.update_params(params, k_size_override=None)

@torch.no_grad()
def run_intrinsic_evo(num_runs=150, default=True):
    for k in tqdm(range(num_runs)):
        dead = False
        new_params(model, default=default)
        model.set_init_circle(radius=model.k_size*4)
        init_mass = model.state.sum()
        dead_mass = 0.06*init_mass
        print('Initial mass:', init_mass)
        print('Dead mass:', dead_mass)
        for i in range(10000):
            model.step(sense_food=True)
            if(i%100==0):
                if(model.state.sum()<dead_mass):
                    print('Model has died')
                    dead=True
                    break

        if(not dead):
            model._save_with_state(path=f'./evoxchan{'default' if default else 'random'}')
            print('Model has survived with ', model.state.sum())

run_intrinsic_evo(num_runs=20, default=True)
run_intrinsic_evo(num_runs=20, default=False)