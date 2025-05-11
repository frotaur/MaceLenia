"""
This script will run intrinsic evolution on the MaCELenia and MaCELeniaCrossChannel models,
saving the parameters of the worlds that survived 10k steps. Then, it will re run the model
and record the 10k steps as a 'timelapse' sped up video. Feel free to modify the parameters,
and adjust the experiments to your liking.
"""

from modules.intrinsic_utility import run_intrinsic_evo, generate_intrisic_videos
from modules.models import MaCELenia, MaCELeniaCrossChannel

INIT_RADIUS = 3.0  # Initial radius of the circle
BETA = 7.0  # Beta value for the model
SENSE = False  # Whether to use the sense food parameter or not
DEFAULT = True  # Whether to use default param gen, otherwise its random param gen
NUM_RECORD = 1000  # Number of steps to record for the video

NUM_RUNS = 1  # Number of runs to perform

#### --------------------------- RUN INTRINSIC EVOLUTION AND SAVE SURVIVORS, DOES IT FOR CROSS CHANNEL AND NORMAL MACELENIA --------------------------- #####
model = MaCELenia(size=(1, 800, 800), dt=0.1, num_channels=3, device="cuda", has_food=True, sense_food=SENSE)

save_path = run_intrinsic_evo(
    model, num_runs=NUM_RUNS, default=DEFAULT, beta=BETA, sense=SENSE, init_radius=INIT_RADIUS
)

model = MaCELeniaCrossChannel(
    size=(1, 800, 800), dt=0.1, num_channels=3, device="cuda", has_food=True, sense_food=SENSE
)

save_path_cross = run_intrinsic_evo(
    model,
    num_runs=NUM_RUNS,
    default=DEFAULT,
    beta=BETA,
    sense=SENSE,
    init_radius=INIT_RADIUS,
    extra_string="cross",
)


#### --------------------------- RUN INTRINSIC EVO AND RECORD FROM SAVED PARAMETERS --------------------------- #####
model = MaCELenia(size=(1, 800, 800), dt=0.1, num_channels=3, device="cuda", has_food=True)
generate_intrisic_videos(
    model=model, saves_folder=save_path, beta=BETA, init_radius=INIT_RADIUS, num_steps=NUM_RECORD, sense=SENSE
)

model = MaCELeniaCrossChannel(size=(1, 800, 800), dt=0.1, num_channels=3, device="cuda", has_food=True)
generate_intrisic_videos(
    model=model,
    saves_folder=save_path_cross,
    beta=BETA,
    init_radius=INIT_RADIUS,
    num_steps=NUM_RECORD,
    sense=SENSE,
)
