import pandas
import pandas as pd

data = pd.read_csv('data/v2.csv')
data = data.dropna(axis=1, how='any')
length = len(data)
train_len = int(0.8*length)
test_len = int(0.1*length)
val_len = length-train_len-test_len
train_data = data.iloc[0:train_len]
test_data = data.iloc[train_len:train_len+test_len]
val_data = data.iloc[train_len+test_len:]

train_data.to_csv('data/train_1.csv')
test_data.to_csv('data/test_1.csv')
val_data.to_csv('data/val_1.csv')