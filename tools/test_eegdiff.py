from EEG import EEGDiffMR
from mmengine import Config

config_file_path = 'config/ventialtion_v1/evaler.py'

config = Config.fromfile(config_file_path)

evaler = EEGDiffMR.build(config.evaler)

evaler.eval()