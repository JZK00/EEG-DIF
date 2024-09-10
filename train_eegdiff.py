from EEG import EEGDiffTrainner,EEGDiffMR,EEGDiffDR
from mmengine import Registry,Config
import wandb

config_file_path = 'config/EEG-Diff/trainner.py'

config = Config.fromfile(config_file_path)

trainner = EEGDiffMR.build(config.trainner)

key = 'a1c9d18c29f582d1f28a8e85102cf731a687817a' ## Fill your key here!

assert key is not None, "please fill in your wandb key"

wandb.login(key=key) 
wandb.init(
    # set the wandb project where this run will be logged
    project=config.project_name,
    name='EEG-diff base', #name here
    # track hyperparameters and run metadata
    config=config.wandb_config
)

trainner.train()
