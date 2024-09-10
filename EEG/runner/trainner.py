import random

from mmengine import Config
from ..registry import EEGDiffMR, EEGDiffDR
from torchvision.transforms import Compose, Normalize
from ..pipeline import DDIMPipeline
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

@EEGDiffMR.register_module()
class EEGDiffTrainner:
    def __init__(self,
                 trainner_config: Config,
                 unet: Config,
                 noise_scheduler: Config,
                 optimizer: Config,
                 train_dataset: Config,
                 val_dataset: Config):
        self.config = trainner_config
        self.unet = EEGDiffMR.build(unet)
        self.initial_unet()
        self.noise_scheduler = EEGDiffMR.build(noise_scheduler)
        self.pipeline = DDIMPipeline(unet=self.unet, scheduler=self.noise_scheduler)
        transform = Compose([
            Normalize(mean=[0.5], std=[0.5]),
        ])
        self.train_dataset = EEGDiffDR.build(train_dataset)
        self.val_dataset = EEGDiffDR.build(val_dataset)
        self.train_dataset.transform = transform
        self.val_dataset.transform = transform
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.eval_batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=optimizer.learning_rate)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.config.num_epochs),
        )

    def initial_unet(self):
        self.unet.to(self.config.device)
        if self.config.u_net_weight_path is not None:
            self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
            print("Load u_net weight from {}".format(self.config.u_net_weight_path))
        else:
            print("No u_net weight path is provided, use random weight")

    def evaluate(self, batch, epoch):
        original_image = batch[0].to(self.config.device)
        father_path = f"{self.config.output_dir}/{epoch}"
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        predicted_image = self.pipeline(
            original_image,
            self.config.prediction_point,
            batch_size=len(original_image),
            num_inference_steps=self.config.num_train_timesteps,
            output_type='numpy'
        ).images
        original_image = (original_image / 2 + 0.5).clamp(0, 1)
        original_image = original_image.cpu().permute(0, 2, 3, 1).numpy()
        diff = np.sum((original_image - predicted_image) ** 2)
        wandb.log({'image diff': diff})

        index = np.random.randint(0, len(original_image) - 1)
        plt.imshow(predicted_image[index, :, :, 0], interpolation='nearest')
        plt.clim(0, 1)
        plt.colorbar()
        plt.savefig(f"{father_path}/predicted_image.png")
        plt.close()
        plt.imshow(original_image[index, :, :, 0], interpolation='nearest')
        plt.clim(0, 1)
        plt.colorbar()
        plt.savefig(f"{father_path}/original_image.png")
        plt.close()
        torch.save(self.unet.state_dict(), f"{father_path}/unet.pth")

    def train_single_batch(self, batch, epoch):
        clean_images = batch[0].to(self.config.device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps,
                                  (clean_images.shape[0],),
                                  device=clean_images.device).long()
        image_add_noise = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        image_add_noise[:, :, 0:self.config.prediction_point, :] = clean_images[:, :, 0:self.config.prediction_point, :]
        noise_pred = self.unet(image_add_noise, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred[:, :, self.config.prediction_point:, :],
                          noise[:, :, self.config.prediction_point:, :])
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        logs = {"epoch": (epoch // len(self.train_dataloader)),
                "iteration": epoch,
                "mse loss": loss.detach().item(),
                "lr": self.lr_scheduler.get_last_lr()[0]}
        print(", ".join([key + ": " + str(round(value, 5)) for key, value in logs.items()]))
        wandb.log(logs)
        return loss.item()

    def train(self):
        number_interation = 0
        for epoch in range(self.config.num_epochs):
            for iteration, batch in enumerate(self.train_dataloader):
                self.train_single_batch(batch, number_interation)
                number_interation+=1
                #print(len(self.val_dataloader.dataset))   #
                #'''
                if number_interation>=self.config.eval_begin and number_interation%self.config.eval_interval==0:
                    index = random.randint(1,len(self.val_dataloader)-1)
                    for eval_iteration,eval_batch in enumerate(self.val_dataloader):
                        if eval_iteration==index:
                            self.evaluate(eval_batch,number_interation)
                            break #'''