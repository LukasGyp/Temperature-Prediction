import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import glob

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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

df = pd.read_csv('data.csv')
params = json.load(open('params.json', 'r'))

cities = params['cities']
feats = params['features']

features = []
for f in feats:
  for c in cities:
    features.append(f'{f}_{c}')

n_features = len(features)

x = df[features].values

train_rate = params['train_rate']
train_size = int(len(x) * train_rate)
test_size = len(x) - train_size

x_train = x[-train_size:]
x_test = x[:test_size]

ms = MinMaxScaler()
x_train = x_train.reshape(-1, n_features)
x_test = x_test.reshape(-1, n_features)

x_train_ms = ms.fit_transform(x_train)
x_test_ms = ms.transform(x_test)

time_step = params['time_step']
BATCH_SIZE = params['BATCH_SIZE']

n_sample_train = train_size - time_step
n_sample_test = test_size - time_step
input_data_train = np.zeros((n_sample_train, time_step, n_features))
correct_data_train = np.zeros((n_sample_train, n_features))
input_data_test = np.zeros((n_sample_test, time_step, n_features))
correct_data_test = np.zeros((n_sample_test, n_features))

for i in range(n_sample_train):
  input_data_train[i] = x_train_ms[i:i+time_step]
  correct_data_train[i] = x_train_ms[i+time_step]

for i in range(n_sample_test):
  input_data_test[i] = x_test_ms[i:i+time_step]
  correct_data_test[i] = x_test_ms[i+time_step]

input_data_train_tensor = torch.tensor(input_data_train, dtype=torch.float)
correct_data_train_tensor = torch.tensor(correct_data_train, dtype=torch.float)
input_data_test_tensor = torch.tensor(input_data_test, dtype=torch.float)
correct_data_test_tensor = torch.tensor(correct_data_test, dtype=torch.float)

dataset = TensorDataset(input_data_train_tensor, correct_data_train_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset_test = TensorDataset(input_data_test_tensor, correct_data_test_tensor)
loader_test = DataLoader(dataset_test, batch_size=16384)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_inputs = n_features
n_outputs = n_features
n_hidden = params["n_hidden"]
n_layers = params["n_layers"]
LEARNING_RATE = params["LEARNING_RATE"]

model = Model(n_inputs, n_outputs, n_hidden, n_layers).to(device)
loss_fnc = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 175, 200, 225, 250, 275], gamma=0.5)

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
ax.set_ylim(0, 0.001)

model_num = len(glob.glob('./model*/')) + 1
dir_name = f'./model_{model_num}'
os.mkdir(dir_name)

model_path = f'./model_{model_num}/model.pth'
torch.save(model.state_dict(), model_path)

fig_path = f'./model_{model_num}/loss.jpg'
plt.savefig(fig_path, dpi=100)

json_path = f'./model_{model_num}/detail.json'
with open(json_path, 'w') as f:
  json.dump(params, f, indent=2)