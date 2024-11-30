# EEG-DIF

## Introduction
Early Warning of Epileptic Seizures through Generative Diffusion Model-based Multi-channel EEG Signals Forecasting. 

Citation: Jiang Z, Dai W, Wei Q, et al. EEG-DIF: Early Warning of Epileptic Seizures through Generative Diffusion Model-based Multi-channel EEG Signals Forecasting[J]. arXiv preprint arXiv:2410.17343, 2024. https://arxiv.org/abs/2410.17343

Here, we provide a clean version of the EEG-DIF algorithm, allowing users to run the code on any EEG data to achieve future predictions for any time frame. This research is ongoing, and additional updates or code uploads will be made in the future.

The EEG-DIF algorithm:
![本地图片描述](images/Fig1-new.bmp)

## Dataset
Siena Scalp EEG Database (PhysioNet). You can download the datasets from https://physionet.org/content/siena-scalp-eeg/1.0.0/.

## Installation
This project is based on Pytorch.
You can use the following command to install the Pytorch.

### Create a virtual enviroment
```bash
conda create -n EEGDiff python=3.9
conda activate EEGDiff
```

linux user
```bash
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

windows user
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

MacOS user
```bash
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

## Forecasting Model Training
You should fill your wandb key in:
```bash
/train_eegdiff.py
```

Then, you can use the following command to train your model.
```bash
python train_eegdiff.py
```

## Forecasting Model Inference
You can use the following command to test your model.
```bash
python test_eegdiff.py
```
## Forecasting Results
You can see some prediction results by using this code, like:
![本地图片描述](images/Fig2-new.bmp)

And you can also calculate the corresponding prediction metrics. You can save the generated future data locally and then run the classification code to carry out early diagnosis and warning of related diseases. Our code is an initial version that offers a novel solution and approach, and we welcome everyone to further improve and enhance it based on this foundation.
