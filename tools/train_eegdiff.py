from EEG import DVMTrainner,EEGDiffMR,EEGDiffDR
from mmengine import Registry,Config
import wandb

config_file_path = 'config/EEG-Diff/trainner.py'

config = Config.fromfile(config_file_path)

trainner = EEGDiffMR.build(config.trainner)

wandb.login(key=None) #key   using your wandb KEY!
wandb.init(
    # set the wandb project where this run will be logged
    project=config.project_name,
    name='train_eegm',
    # track hyperparameters and run metadata
    config=config.wandb_config
)

trainner.train()
