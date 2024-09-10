from EEG import EEGDataset,evaluationDataset

train_dataset = dict(
    type= evaluationDataset, #EEGDataset,
    csv_path="data/SCD_train.csv", 
)

val_dataset = dict(
    type=EEGDataset,
    csv_path="data/SCD_train.csv",
)