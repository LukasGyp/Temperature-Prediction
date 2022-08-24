import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
import glob

from pickle import dump
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. モデルの構築
## CNN + LSTM
## Conv2d: input(BATCH_SIZE, n_channels, heights, widths), output(BATCH_SIZE, n_channels_out, height_out, width_out)
## x: (BATCH_SIZE, n_features, time_step, n_cities)
## y: (BATCH_SIZE, n_channels_out, H, W)
## MaxPool: input(N, C, H, W) output(N, C, Ho, Wo)
## LSTM: input(N, L, H), output(N, L, H)

class Model(nn.Module):
  def __init__(self, in_channels, n_cities, output_size, hidden_dim, n_layers, n_channels1=32, n_channels2=64):
    super(Model, self).__init__()

    self.cnn1 = nn.Sequential(
      # BS * nf * ts * nc
      nn.Conv2d(in_channels, n_channels1, kernel_size=3, padding=1),
      # BS * 32 * ts * nc
      nn.BatchNorm2d(n_channels1),
      nn.ReLU(),
      nn.MaxPool2d(2)
      # BS * 32 * ts/2 * nc/2
    )
    self.cnn2 = nn.Sequential(
      # BS * 32 * ts/2 * nc/2
      nn.Conv2d(n_channels1, n_channels2, kernel_size=3, padding=1),
      # BS * 64 * ts/2 * nc/2
      nn.BatchNorm2d(n_channels2),
      nn.ReLU()
      # BS * 64 * ts/2 * nc/2
    )
    # BS * ts/2 * (64 * nc/2)
    self.lstm = nn.LSTM(int(n_channels2*n_cities/2), hidden_dim, n_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_size)
  
  def forward(self, x):
    x = x.to(device)
    y_cnn1 = self.cnn1(x)
    y_cnn2 = self.cnn2(y_cnn1)
    #次元変える
    lstm_input = torch.concat([y_cnn2[:, i, :, :] for i in range(y_cnn2.shape[1])], axis=2)
    y_lstm, h = self.lstm(lstm_input, None)
    y = self.fc(y_lstm[:, -1, :])
    return y

df = pd.read_csv('data.csv')
params = json.load(open('params.json', 'r'))

cities = params['cities']
feats = params['features']

n_cities = len(cities)
n_features = len(feats)

all_features = []
for f in feats:
  for c in cities:
    all_features.append(f'{f}_{c}')
all_n_features = n_features * n_cities

city_feature_ = []
for c in cities:
  city_feature = []
  for f in feats:
    city_feature.append(f'{f}_{c}')
  city_feature_.append(city_feature)

#city * time * feature
data = np.empty([0, df.shape[0], n_features])
for city in city_feature_:
  data = np.append(data, df.loc[:, city].values.reshape(1, df.shape[0], n_features), axis=0)


train_rate = params['train_rate']
train_size = int(data.shape[1] * train_rate)
test_size = data.shape[1] - train_size

x_train = data[:, -train_size:, :]
x_test = data[:, :test_size, :]

ms = MinMaxScaler()
x_train_ms = ms.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_test_ms = ms.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

#feature * time * city
x_train_ms = np.transpose(x_train_ms, (2, 1, 0))
x_test_ms = np.transpose(x_test_ms, (2, 1, 0))

time_step = params['time_step']
BATCH_SIZE = params['BATCH_SIZE']

n_sample_train = train_size - time_step
n_sample_test = test_size - time_step
input_data_train = np.zeros((n_sample_train, n_features, time_step, n_cities))
correct_data_train = np.zeros((n_sample_train, all_n_features))
input_data_test = np.zeros((n_sample_test, n_features, time_step, n_cities))
correct_data_test = np.zeros((n_sample_test, all_n_features))

# correct: time * (feat * city)
data_for_correct = np.transpose(x_train_ms, (1, 0, 2)).reshape(-1, all_n_features)
for i in range(n_sample_train):
  input_data_train[i] = x_train_ms[:, i:i+time_step, :]
  correct_data_train[i] = data_for_correct[i+time_step, :]

data_for_correct = np.transpose(x_test_ms, (1, 0, 2)).reshape(-1, all_n_features)
for i in range(n_sample_test):
  input_data_test[i] = x_test_ms[:, i:i+time_step, :]
  correct_data_test[i] = data_for_correct[i+time_step, :]

input_data_train_tensor = torch.tensor(input_data_train, dtype=torch.float)
correct_data_train_tensor = torch.tensor(correct_data_train, dtype=torch.float)
input_data_test_tensor = torch.tensor(input_data_test, dtype=torch.float)
correct_data_test_tensor = torch.tensor(correct_data_test, dtype=torch.float)

dataset = TensorDataset(input_data_train_tensor, correct_data_train_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset_test = TensorDataset(input_data_test_tensor, correct_data_test_tensor)
loader_test = DataLoader(dataset_test, batch_size=16384)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_channels = n_features
n_hidden = params["n_hidden"]
n_layers = params["n_layers"]
LEARNING_RATE = params["LEARNING_RATE"]

model = Model(in_channels=n_channels, output_size=all_n_features, n_cities=n_cities, hidden_dim=n_hidden, n_layers=n_layers).to(device)
loss_fnc = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 175, 200, 225, 250, 275], gamma=0.5)

print(model)

start = time.time()
EPOCHS = params['EPOCHS']
record_loss_train = []
record_loss_test = []

print(f'Using device: {device}')

for i in range(1, EPOCHS+1):
  model.train()
  loss_train = 0
  for j, (x, t) in enumerate(loader):
    pred_y = model(x)
    loss = loss_fnc(pred_y, t.to(device))
    loss_train += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  loss_train /= j+1
  record_loss_train.append(loss_train)

  model.eval()
  loss_test = 0
  for j, (x, t) in enumerate(loader_test):
    with torch.no_grad():
      pred_y = model(x)
      loss = loss_fnc(pred_y, t.to(device))
      loss_test += loss.item()
  loss_test /= j+1  
  record_loss_test.append(loss_test)

  if i % 1 == 0:
    print(f"Epoch: {i}\tLoss_Train: {loss_train:.7f}\tLoss_Test: {loss_test:.7f}")
  scheduler.step()

elapsed_time = time.time() - start
print(f'Training finished. Total time: {elapsed_time:.2f} [s]')
params['Train Time'] = elapsed_time
params['Train Loss'] = record_loss_train[-1]
params['Test Loss'] = record_loss_test[-1]


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(record_loss_train)), record_loss_train,label="train loss")
ax.plot(range(len(record_loss_test)), record_loss_test, label="test loss")
ax.legend()
ax.set_xlabel("Epochs")
ax.set_ylabel("Error")
ax.set_ylim(record_loss_train[-1]*0.7, record_loss_test[-1]*1.5)

model_num = len(glob.glob('./model*/')) + 1
dir_name = f'./model_{model_num}/'
Path(dir_name).mkdir(parents=True, exist_ok=True)

model_path = f'./model_{model_num}/model.pth'
torch.save(model.state_dict(), model_path)

scaler_path = f'model_{model_num}/scaler.pkl'
dump(ms, open(scaler_path, "wb"))

fig_path = f'./model_{model_num}/loss.jpg'
plt.savefig(fig_path, dpi=100)

json_path = f'./model_{model_num}/detail.json'
with open(json_path, 'w') as f:
  json.dump(params, f, indent=2)