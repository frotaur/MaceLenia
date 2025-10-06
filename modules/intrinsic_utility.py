"""Script used to run intrinsic evolution experiments on MaceLenia"""

from modules import MaCELenia, MaCELeniaCrossChannel
from modules.models.utils import LeniaParams
from tqdm import tqdm
import torch
from pathlib import Path
from modules.main_utils import launch_video, add_frame


def new_params(model, default=True):
    """
    Generates new random parameters for the model.
    Can be modified at will to explore different parameter regions.
    """
    if not default:
        params = LeniaParams.mixed_gen(
            batch_size=model.batch,
            num_channels=model.C,
            device=model.device,
            k_size=model.k_size,
            k_mult=model.k_mult,
            k_arbi=True,
            g_arbi=False,
        )
    else:
        params = LeniaParams.default_gen(
            batch_size=model.batch,
            num_channels=model.C,
            device=model.device,
            k_size=model.k_size,
            k_mult=model.k_mult,
        )
    model.update_params(params, k_size_override=None)


@torch.no_grad()
def run_intrinsic_evo(model: MaCELenia, beta=7., init_radius=3., num_runs=150, default=True, sense=True, extra_string="",
                      param_save_folder="./intrinsic_evo"):
    """
    Runs an intrinsic evolution run, with the specified model and parameters. Feel free to modify at
    will, lots to explore here. A lot of parameters are hardcoded in the body.

    Args:
        model (MaCELenia): The model to run the evolution on.
        num_runs (int): The number of runs to perform. (each run is 10k steps)
        default (bool): Whether to use default parameters or random ones.
        sense (bool): Whether to use the sense food parameter or not.
        extra_string (str): Extra string to append to the save path.
    
        
    returns:
        str: The path to the folder where the parameters are saved.
    """
    param_save_folder = Path(f"{param_save_folder}_{"sense" if sense else "nosense"}{f'_{extra_string}' if extra_string!= '' else ''}")
    param_save_folder.mkdir(parents=True, exist_ok=True)

    for k in tqdm(range(num_runs)):
        dead = False
        new_params(model, default=default)  # Generate and update model parameters

        model.set_init_circle(radius=model.k_size * init_radius)  # Initialize to a circle (can modify radius)
        model.b = beta  # Set Beta to 6.0 (can modify)
        model.sense_food = sense  # Set sense food to True or False (can modify)
        init_mass = model.state.sum()
        dead_mass = (
            0.1 * init_mass  # Define the model as dead if it retains less than 10% of its initial mass (can modify for how aggressive you want the evolution to be)
        )

        for i in range(10000):  # Simulate for 10k steps (can modify)
            model.step()
            if i % 100 == 0:
                if model.state.sum() < dead_mass:  # If the model is already dead, cut the simulation short
                    print("Model has died")
                    dead = True
                    break

        if not dead:  # Save the final state if the model is still alive
            model._save_with_state(path=param_save_folder)
            print("Model has survived with ", model.state.sum())

    return param_save_folder.as_posix()  # Return the path to the folder with the saved parameters


@torch.no_grad()
def generate_intrisic_videos(model: MaCELenia, saves_folder, beta=7., init_radius=3.,record_every=5, num_steps=10000, fps=60,sense=True, out_video_folder = 'intrinsic_timelapse'):
    """
        Generate 'timelapse' like videos of the intrinsic evolution process.

        Args:
            model (MaCELenia): The model to run the evolution on.
            saves_folder (str): The folder where the .pt files are saved.
            record_every (int): How often to record a frame.
            num_steps (int): Number of steps to simulate.
            fps (int): Frames per second for the video.
            sense (bool): Whether to allow food sensing
    """

    # Collect all .pt files in the save_folder
    pt_files = list(Path(saves_folder).glob('**/*.pt'))
    if not pt_files:
        print(f"No .pt files found in {saves_folder}")
        return

    # Now we have a list of Path objects with the full paths to all .pt files
    print(f"Found {len(pt_files)} parameter files")

    save_folder = Path(out_video_folder)
    save_folder.mkdir(parents=True, exist_ok=True)  # Create the save folder if it doesn't exist

    print(f"Generating videos in {save_folder}")

    for param_path in tqdm(pt_files):
        param = LeniaParams(from_file=param_path, device=model.device)
        model.update_params(param, k_size_override=None)

        model.set_init_circle(radius=model.k_size * init_radius)  # Initialize to a circle (can modify radius)
        model.b = beta  # Set Beta to 6.0 (can modify)

        videowriter = launch_video(
            size=(model.h, model.w),
            fps=fps,
            save_folder=save_folder,
            save_name=param.name,
            fourcc="mp4v",
        )

        for i in range(num_steps):  # Simulate for specified steps (can modify)
            model.step()
            if i % record_every == 0:
                model.draw()
                add_frame(videowriter, model.worldsurface)

        videowriter.release()
        