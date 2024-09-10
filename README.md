# EEG-DIF

## Introduction

## Installation
This project is based on Pytorch.
You can use the following command to install the Pytorch.

### create a virtual enviroment
```bash
conda create -n EEGDiff python=3.9
conda activate EEGDiff
```

windows user
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

linux user
```bash
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

MacOS user
```bash
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

## Model Trainning
You can use the following command to train the model.
```bash
python tools/train_eegdiff.py
```

## Model Testing
You can use the following command to test the model.
```bash
python tools/test_eegdiff.py
```


## Weight Download
first download the weight file.
```bash
python tools/download_weight.py
```
or you can download through this google drive link and move it to the weight folder.
[google drive link](https://drive.google.com/file/d/1eW55eq7pHaBEba99B7svK_tAL9yRy36q/view?usp=sharing)
