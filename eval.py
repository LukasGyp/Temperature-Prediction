import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pickle import load
import os

import torch
import torch.nn as nn

model_num = int(input('Model number to evaluete: '))

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

train_rate = input_params['train_rate']
train_size = int(len(x) * train_rate)
test_size = len(x) - train_size

x_test = x[:test_size]

scaler_path = f'model_{model_num}/scaler.pkl'
ms = load(open(scaler_path, "rb"))
x_test = x_test.reshape(-1, n_features)

x_test_ms = ms.transform(x_test)

time_step = input_params['time_step']

n_sample = test_size - time_step
input_data = np.zeros((n_sample, time_step, n_features))
correct_data = np.zeros((n_sample, n_features))

for i in range(n_sample):
  input_data[i] = x_test_ms[i:i+time_step]
  correct_data[i] = x_test_ms[i+time_step]

input_data_tensor = torch.tensor(input_data, dtype=torch.float)
correct_data_tensor = torch.tensor(correct_data, dtype=torch.float)

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

def loss_fnc(pred_y, y):
  return np.abs(pred_y - y)

model.eval()
eval_hour = 24
total_loss = 0
loss_data = np.empty((n_features, 0, eval_hour+1))
with torch.no_grad():
  for i in range(n_sample-eval_hour):
    loss_ = np.zeros((n_features, 1, 1))
    predicted = input_data_tensor[i].numpy()
    for h in range(eval_hour):
      x_input = torch.tensor(predicted[-time_step:], dtype=torch.float)
      x_input = x_input.reshape(1, time_step, n_features)
      pred_y = model(x_input)
      predicted = np.append(predicted, pred_y.cpu(), axis=0)
      t = ms.inverse_transform(correct_data_tensor[i+h].reshape(-1, n_features))
      pred_y = ms.inverse_transform(pred_y.cpu().reshape(-1, n_features))
      loss = loss_fnc(pred_y, t)
      loss = loss.reshape(n_features, 1, 1)
      loss_ = np.append(loss_, loss, axis=2)
    loss_data = np.append(loss_data, loss_, axis=1)

loss_dfs = [pd.DataFrame(loss_data[i]) for i in range(n_features)]

for i in range(n_features):
  loss_df = loss_dfs[i]
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.plot(range(1, eval_hour+2), loss_df.mean(), label='average error')
  ax.boxplot(loss_df, whis=1000, labels=range(0, eval_hour+1))
  ax.set_xlabel('hours')
  ax.set_ylabel('error')
  ax.set_yticks(range(0, 16))
  ax.grid(axis='y')
  ax.legend()
  ax.set_title(features[i])
  os.mkdir('time_loss')
  filename = f'model_{model_num}/time_loss/{features[i]}.jpg'
  plt.savefig(filename, dpi=100)

  os.mkdir('loss_data')
  loss_data_path = f'model_{model_num}/loss_data/{features[i]}.csv'
  loss_df.describe().to_csv(loss_data_path)
