from EEG import EEGDiffTrainner,EEGDiffMR,EEGDiffDR
from mmengine import Registry,Config
import wandb

config_file_path = 'config/EEG-Diff/trainner.py'

config = Config.fromfile(config_file_path)

trainner = EEGDiffMR.build(config.trainner)

key = None #Fill your key here

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
