import pandas as pd

cities = [
  'sapporo',
  'aomori',
  'sendai',
  'nigata',
  'tokyo',
  'fukui',
  'nagoya',
  'osaka',
  'matsue',
  'matsuyama',
  'fukuoka',
  'kagoshima',
  'naha',
]

features = [
  'Atm1',
  'Atm2',
  'Rain',
  'Temp',
  'DP',
  'VP',
  'Humidity',
  'Sun Shine',
  'Insolation',
  'Wind E',
  'Wind N'
]

dfs = []

for city in cities:
  filename = f'data/{city}.csv'
  rename_dic = {}
  for f in features:
    new_feature = f'{f}_{city}'
    rename_dic[f] = new_feature

  df_local = pd.read_csv(filename)
  df_local = df_local.rename(columns=rename_dic)
  dfs.append(df_local)

df = dfs[0]
for i in range(1, len(cities)):
  df = df.merge(dfs[i], on=['Year', 'Month', 'Day', 'Hour'])

df.to_csv('data.csv', index=False)
