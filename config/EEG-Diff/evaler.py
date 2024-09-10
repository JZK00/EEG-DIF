from mmengine import read_base
from EEG import EEGDiffEvaler,Long_predictionEEGDataset,evaluationDataset

with read_base():
    from ._base_.train_dataset import train_dataset
    from ._base_.unet import unet
    from ._base_.noise_scheduler import noise_scheduler
    from ._base_.basic_information import prject_name

device = "cuda"

dataset = dict(
    type=evaluationDataset,
    csv_path='data/signals_r01.csv',  ## here, 没用
)

eval_config = dict(
    device=device,
    csv_path='data/SCD_test.csv', ## eval set!!! 测试数据 
    output_dir="caches/ventilation_v1/unet",  ## 没用
    u_net_weight_path = 'caches/mitbih_af_v2/unet1/30000/unet.pth', ## Change it
    prediction_point=8,  #Sampler窗口每次移动 window_size - num_prediction_point, here is 16-8=8.
    num_train_timesteps=200,
    plot_shifted = False,
    window_size=16,  ##新加
    ###########################################################################################
    batch_size=1, ## Obsolete
    batch_index=0,
)

project_name = prject_name

evaler = dict(
    type=EEGDiffEvaler,
    evaler_config=eval_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
    dataset=train_dataset,)
