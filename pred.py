import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pickle import load

import torch
import torch.nn as nn

model_num = int(input('Model number to predict: '))

df = pd.read_csv('data.csv')
json_path = f'model_{model_num}/detail.json'
input_params = json.load(open(json_path, 'r'))

cities = input_params['cities']
feats = input_params['features']

features = []
for f in feats:
  for c in cities:
    features.append(f'{f}_{c}')

n_features = len(features)

x = df[features].values

time_step = input_params['time_step']
x = x[-time_step:]

scaler_path = f'model_{model_num}/scaler.pkl'
ms = load(open(scaler_path, "rb"))
x = x.reshape(-1, n_features)
x_ms = ms.transform(x)

n_sample = 1
input_data = np.zeros((n_sample, time_step, n_features))

input_data[0] = x_ms[-time_step:]

input_data_tensor = torch.tensor(input_data, dtype=torch.float)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_inputs = n_features
n_outputs = n_features
n_hidden = input_params["n_hidden"]
n_layers = input_params["n_layers"]
LEARNING_RATE = input_params["LEARNING_RATE"]

class Model(nn.Module):
  def __init__(self, input_size, output_size, hidden_dim, n_layers):
    super(Model, self).__init__()

    self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_size)
  
  def forward(self, x):
    x = x.to(device)
    y_lstm, h = self.lstm(x, None)
    y = self.fc(y_lstm[:, -1, :])
    return y

model = Model(n_inputs, n_outputs, n_hidden, n_layers).to(device)

model_path = f'model_{model_num}/model.pth'
model.load_state_dict(torch.load(model_path))

model.eval()
eval_hour = 12
with torch.no_grad():
  predicted = input_data_tensor.numpy()
  for h in range(eval_hour):
    x_input = torch.tensor(predicted[:, -time_step:], dtype=torch.float)
    x_input = x_input.reshape(1, time_step, n_features)
    pred_y = model(x_input)
    predicted = np.append(predicted, pred_y.cpu().reshape(1, 1, n_features), axis=1)
  predicted = ms.inverse_transform(predicted.reshape(-1, n_features))
print(predicted)

for i in range(n_features):
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.plot(range(-time_step, 1), predicted[:time_step+1, i])
  ax.plot(range(0, eval_hour+1), predicted[time_step:, i])
  ax.set_xlabel("Hours")
  ax.set_ylabel("Predictions")
  figname = f'model_{model_num}/pred_{features[i]}.jpg'
  plt.save(figname, dpi=100)