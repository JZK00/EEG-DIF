from EEG import UNet2DModel

unet = dict(
    type=UNet2DModel,
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    block_out_channels=(224, 448, 672, 896),
    in_channels=1,
    out_channels=1,
    sample_size=(16, 16),  ## here, 56*56 or 30*30 or others
)
