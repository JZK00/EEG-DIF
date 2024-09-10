import wandb
run = wandb.init()
artifact = run.use_artifact('EEG-DIF:v0', type='model')  ## wait for updating
artifact_dir = artifact.download()
