from mmengine import Registry
from mmengine import MODELS
from mmengine import DATASETS

EEGDiffMR = Registry('EEGDiffMR', parent=MODELS)
EEGDiffDR = Registry('EEGDiffDR', parent=DATASETS)
