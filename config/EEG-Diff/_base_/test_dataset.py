from EEG import EEGDataset

test_dataset = dict(
    type=EEGDataset,
    csv_path="data/0310test.csv",  #不对，去掉即可
)

# val_dataset = dict(
#     type=VentilationDataset,
#     csv_path="data/val_1.csv",
# )