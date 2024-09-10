from mmengine import Config
from ..registry import EEGDiffMR, EEGDiffDR
from torchvision.transforms import Compose, Normalize
from ..pipeline import DDIMPipeline
from torch.utils.data import DataLoader,Sampler
from ..dataset.eval_dataset import evaluationDataset
import torch
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
from ..utils import compare
from tqdm import tqdm



@EEGDiffMR.register_module()
class EEGDiffEvaler:
    def __init__(self,
                 unet: Config,
                 dataset: Config,
                 evaler_config: Config,
                 noise_scheduler: Config):
        self.config = evaler_config
        self.unet = EEGDiffMR.build(unet)
        self.initial_unet()
        self.noise_scheduler = EEGDiffMR.build(noise_scheduler)
        #self.dataset = DVDR.build(dataset)
        self.dataset = evaluationDataset(csv_path=self.config.csv_path,window_size=self.config.window_size,step_size=self.config.window_size-self.config.prediction_point)  ##改动
        data_title_string = "Temp,Humidity,PM1.0,PM2.5,PM10,CO2,Formaldehyde,TVOC,FVC	FEV1	PEF	PM1.0	PM2.5	PM10	CO2	Temp°C" #,F16,F17,F18,F19,F20,F21,F22,F23,F24,F25,F26,F27,F28,F29,F30
        #data_title_string = "Middle_Temp,Middle-Humidity,PM1.0,PM2.5,PM10,CO2,Formaldehyde,TVOC,FVCL,FEV1L,FEV1FVC,FEF25,FEF50,FEF75,FEF25-75,PEFL,VCMAXL,VTL,ERVL,IRVL,ICL,VCINL,VCEXL,VCEXL"
        self.title_list = data_title_string.split(',')
        self.pipeline = DDIMPipeline(unet=self.unet, scheduler=self.noise_scheduler)
        self.batch_size = (len(self.dataset.data)-self.config.window_size)//(self.config.window_size-self.config.prediction_point) + 1
        #sampler = torch.utils.data.SequentialSampler(range(self.config.prediction_point, len(self.dataset)))
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, 
                                     drop_last=(len(self.dataset.data)-self.config.window_size)%(self.config.window_size-self.config.prediction_point) == 0)
        #self.list1 = ["Occupancy", "door_gap", "window_gap", "humidity", "VOC_ppm", "temperature_Main", "outdoor_temperature", "outdoor_windgust", "outdoor_humidity", ]

    def initial_unet(self):
        self.unet.to(self.config.device)
        if self.config.u_net_weight_path is not None:
            self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
            print("Load u_net weight from {}".format(self.config.u_net_weight_path))
        else:
            print("No u_net weight path is provided, use random weight")

    def get_batch(self,index):
        # select desired image to do the test.
        data_iter = iter(self.dataloader)
        desired_index = index
        print(self.dataloader.dataset)
        try:
            for i in range(desired_index + 1):
                batch = next(data_iter)
        except StopIteration:
            print("索引超出了DataLoader中的batch数量。")

        # precess the data to thr prediction
        inputs = batch[0]
        return inputs



    def eval(self):
        inputs = self.get_batch(self.config.batch_index)
        inputs = inputs.to(self.config.device)
        # now the image is half clear image half randon noise.
        print(inputs.shape)
        # use 1 iteration scheme to do the prediction.
        for _ ,input in tqdm(enumerate(inputs),desc="validating"):
            input = input.unsqueeze(0)
            image = randn_tensor(input.shape, device=self.config.device, dtype=self.unet.dtype)
            image[:, :, 0:self.config.prediction_point, :] = input[:, :, 0:self.config.prediction_point, :]
            image = self.pipeline.do_prediction(
                image,
                self.config.prediction_point,
                batch_size=len(input),
                num_inference_steps=self.config.num_train_timesteps,
                output_type='numpy'
            )
            predicted_image = image.clamp(0,1)#(image / 2 + 0.5).clamp(0, 1)
            predicted_image = predicted_image.cpu().permute(0, 2, 3, 1).numpy()
            original_image = input
            original_image = original_image.cpu().permute(0, 2, 3, 1).numpy()
            shifted_original = (input / 2 + 0.5).clamp(0, 1)
            shifted_original = shifted_original.cpu().permute(0, 2, 3, 1).numpy()
            min_number = np.zeros_like(predicted_image)
            max_number = np.zeros_like(predicted_image)

            for i in range(len(predicted_image)):
                for row in range(len(predicted_image[0])):
                    min_number[i, row, :, 0] = self.dataset.min_value
                    max_number[i, row, :, 0] = self.dataset.max_value
            
            predicted_image = predicted_image * (max_number - min_number) + min_number
            original_image = original_image * (max_number - min_number) + min_number
            shifted_original = shifted_original * (max_number - min_number) + min_number

            predicted_image_ = predicted_image[0, :, :, 0]
            original_image_ = original_image[0, :, :, 0]
            shifted_original_ = shifted_original[0, :, :, 0]

            if _ == 0:
                complete_prediction_image = predicted_image_
                complete_original_image = original_image_
                complete_shifted_original = shifted_original_
            else:
                complete_prediction_image = np.vstack((complete_prediction_image,predicted_image_[self.config.prediction_point:,:]))
                complete_original_image = np.vstack((complete_original_image,original_image_[self.config.prediction_point:,:]))
                complete_shifted_original = np.vstack((complete_shifted_original,shifted_original_[self.config.prediction_point:,:]))
        np.save('pred.npy',complete_prediction_image)  ##
        np.save('orig.npy',complete_original_image)    ## 改动
        compare(complete_original_image, complete_prediction_image,complete_shifted_original, self.config.plot_shifted, self.title_list)
       

"""   
# Obsolete
    def ___eval_(self):
        inputs = self.get_batch(self.config.batch_index)
        inputs = inputs.to(self.config.device)
        image = randn_tensor(inputs.shape, device=self.config.device, dtype=self.unet.dtype)
        image[:, :, 0:self.config.prediction_point, :] = inputs[:, :, 0:self.config.prediction_point, :]
        # now the image is half clear image half randon noise.

        # use 1 iteration scheme to do the prediction.
        for i in range(1):
            image = self.pipeline.do_prediction(
                image,
                self.config.prediction_point,
                batch_size=len(inputs),
                num_inference_steps=self.config.num_train_timesteps,
                output_type='numpy'
            )
        # post process the images to make it to 0-1
        predicted_image = (image / 2 + 0.5).clamp(0, 1)
        predicted_image = predicted_image.cpu()#.permute(0, 2, 3, 1).numpy()
        original_image = input #(inputs / 2 + 0.5).clamp(0, 1)
        original_image = original_image.cpu()#.permute(0, 2, 3, 1).numpy()
        
        # recover the value to original value scale
        min_number = np.zeros_like(predicted_image)
        max_number = np.zeros_like(predicted_image)
        for i in range(len(predicted_image)):
            for row in range(len(predicted_image[0])):
                min_number[i, row, :, 0] = self.dataset.min_value
                max_number[i, row, :, 0] = self.dataset.max_value
        predicted_image = predicted_image * (max_number - min_number) + min_number
        original_image = original_image * (max_number - min_number) + min_number
        predicted_image_ = predicted_image[0, :, :, 0]
        original_image_ = original_image[0, :, :, 0]
        compare(original_image_, predicted_image_, self.config.prediction_point, self.title_list)
"""
