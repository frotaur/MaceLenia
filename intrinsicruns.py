""" Script used to run intrinsic evolution experiments on MaceLenia"""

from modules import MaCELenia, MaCELeniaCrossChannel
from modules.models.utils import LeniaParams
from tqdm import tqdm
import torch


def new_params(model, default=True):
    """
        Generates new random parameters for the model.
        Can be modified at will to explore different parameter regions.
    """
    if(not default):
        params = LeniaParams.mixed_gen(
            batch_size=model.batch, num_channels=model.C, device=model.device, k_size=model.k_size,
            k_mult=model.k_mult, k_arbi=True, g_arbi=True
        )
    else:
        params = LeniaParams.default_gen(
            batch_size=model.batch, num_channels=model.C, device=model.device, k_size=model.k_size,
            k_mult=model.k_mult
        )
    model.update_params(params, k_size_override=None)

@torch.no_grad()
def run_intrinsic_evo(model:MaCELenia,num_runs=150, default=True, sense=True,extra_string=''):
    """
        Runs an intrinsic evolution run, with the specified model and parameters. Feel free to modify at 
        will, lots to explore here. A lot of parameters are hardcoded in the body.

        Args:
            model (MaCELenia): The model to run the evolution on.
            num_runs (int): The number of runs to perform. (each run is 10k steps)
            default (bool): Whether to use default parameters or random ones.
            sense (bool): Whether to use the sense food parameter or not.
            extra_string (str): Extra string to append to the save path.
    """

    for k in tqdm(range(num_runs)):
        dead = False
        new_params(model, default=default) # Generate and update model parameters

        model.set_init_circle(radius=model.k_size*3) # Initialize to a circle (can modify radius)
        model.b = 6. # Set Beta to 6.0 (can modify)

        init_mass = model.state.sum()
        dead_mass = 0.06*init_mass # Define the model as dead if it retains less than 6% of its initial mass (can modify for how aggressive you want the evolution to be)

        print('Initial mass:', init_mass)
        print('Dead mass:', dead_mass)

        for i in range(10000): # Simulate for 10k steps (can modify)
            model.step(sense_food=sense)
            if(i%100==0):
                if(model.state.sum()<dead_mass): # If the model is already dead, cut the simulation short
                    print('Model has died')
                    dead=True
                    break

        if(not dead): # Save the final state if the model is still alive
            model._save_with_state(path=f'./evotest_{"sense" if sense else "nosense"}"_{extra_string}"')
            print('Model has survived with ', model.state.sum())

model = MaCELenia(size=(1,800,800), dt=0.1, num_channels=3, device='cuda', has_food=True, save_dir='intrinsic')

run_intrinsic_evo(model,num_runs=30, default=True)
run_intrinsic_evo(model,num_runs=50, default=False)

model = MaCELeniaCrossChannel(size=(1,800,800), dt=0.1, num_channels=3, device='cuda', has_food=True, save_dir='intrinsic')

run_intrinsic_evo(model,num_runs=30, default=True, extra_string='cross')
run_intrinsic_evo(model,num_runs=50, default=False, extra_string='cross')