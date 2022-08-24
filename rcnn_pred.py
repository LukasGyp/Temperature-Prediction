import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pickle import load
from pathlib import Path

import torch
import torch.nn as nn

def to_cnn_shape(x, n_cities, n_features):
  output = np.empty((0, x.shape[0], n_cities))
  for i in range(n_features):
    a = x[:, n_cities*i:n_cities*(i+1)].reshape(1, -1, n_cities)
    output = np.append(output, a, axis=0)
  return output

def to_rnn_shape(x, n_cities, n_features):
  return np.transpose(x, (1, 0, 2)).reshape(-1, n_cities*n_features)

model_num = int(input('Model number to predict: '))

df = pd.read_csv('data.csv')
params = json.load(open('params.json', 'r'))

cities = params['cities']
feats = params['features']
time_step = params['time_step']
BATCH_SIZE = params['BATCH_SIZE']

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

x_train = data[:, -time_step:, :]

scaler_path = f'model_{model_num}/scaler.pkl'
ms = load(open(scaler_path, "rb"))
x_train_ms = ms.transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)

#feature * time * city
x_train_ms = np.transpose(x_train_ms, (2, 1, 0))

input_data_tensor = torch.tensor(x_train_ms, dtype=torch.float)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_inputs = n_features
n_outputs = n_features
n_hidden = params["n_hidden"]
n_layers = params["n_layers"]
LEARNING_RATE = params["LEARNING_RATE"]

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

n_channels = n_features
model = Model(in_channels=n_channels, output_size=all_n_features, n_cities=n_cities, hidden_dim=n_hidden, n_layers=n_layers).to(device)
model_path = f'model_{model_num}/model.pth'
model.load_state_dict(torch.load(model_path))

model.eval()
eval_hour = 72
with torch.no_grad():
  # feature * time * city
  predicted = input_data_tensor.numpy()
  for h in range(eval_hour):
    x_input = torch.tensor(predicted[:, -time_step:, :], dtype=torch.float).reshape(1, n_features, time_step, n_cities)
    pred_y = to_cnn_shape(model(x_input).cpu().numpy(), n_cities, n_features)
    predicted = np.append(predicted, pred_y, axis=1)
  # f * t * c
  predicted = ms.inverse_transform(np.transpose(predicted, (2, 1, 0)).reshape(-1, n_features)).reshape(n_cities, -1, n_features)
  predicted = to_rnn_shape(np.transpose(predicted, (2, 1, 0)), n_cities, n_features)

dir_path = f'model_{model_num}/pred/'
Path(dir_path).mkdir(parents=True, exist_ok=True)

for i in range(all_n_features):
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.plot(range(-time_step+1, 1), predicted[:time_step, i])
  ax.plot(range(0, eval_hour+1), predicted[time_step-1:, i])
  ax.set_xlabel("Hours")
  ax.set_ylabel("Predictions")
  figname = f'model_{model_num}/pred/{all_features[i]}.jpg'
  plt.savefig(figname, dpi=100)
  plt.close()