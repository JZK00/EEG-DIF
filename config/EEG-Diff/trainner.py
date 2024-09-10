from mmengine import read_base
from EEG import EEGDiffTrainner

with read_base():
    from ._base_.train_dataset import train_dataset, val_dataset
    from ._base_.unet import unet
    from ._base_.noise_scheduler import noise_scheduler
    from ._base_.basic_information import prject_name

device = "cuda"
train_config = dict(
    device=device,
    output_dir="caches/EEG_v1/unet1", ##change
    u_net_weight_path=None, #"caches/mitbih_af_v2/unet/122500/unet.pth",
    prediction_point=8,     ## predict nums
    num_train_timesteps=200,
    num_epochs=100,   ##
    train_batch_size=100,
    eval_batch_size=40,
    learning_rate=0.00002, ## 0.00002 or 5
    lr_warmup_steps=5,    ## change it,  10
    eval_begin=5000,   ## 1000 500
    eval_interval=500, ## 500 200
)

project_name = prject_name

trainner = dict(
    type=EEGDiffTrainner,
    trainner_config=train_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
    optimizer=dict(type="Adam", learning_rate=0.00002), ## as the same as
    train_dataset=train_dataset,
    val_dataset=val_dataset)

wandb_config = dict(
    learning_rate=trainner['optimizer']['learning_rate'],
    architecture="diffusion model",
    dataset=project_name,
    epochs=train_config['num_epochs'],
)