import wandb
run = wandb.init()
artifact = run.use_artifact('yue-li22/model-registry/ventilation_diffusion_model:v0', type='model')
artifact_dir = artifact.download()