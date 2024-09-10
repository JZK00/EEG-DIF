# EEG-DIF

## Introduction
Early Warning of Epileptic Seizures through Generative Diffusion Model-based Multi-channel EEG Signals Forecasting.

Here, we provide a clean version of the EEG-DIF algorithm, allowing users to run the code on any EEG data to achieve future predictions for any time frame. This research is ongoing, and additional updates or code uploads will be made in the future.

## Installation
This project is based on Pytorch.
You can use the following command to install the Pytorch.

### create a virtual enviroment
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

## Model Trainning
You can use the following command to train your model.
```bash
python train_eegdiff.py
```

## Model Testing
You can use the following command to test your model.
```bash
python test_eegdiff.py
```
