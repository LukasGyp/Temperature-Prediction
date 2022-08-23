import pandas as pd
import time
import datetime
from bs4 import BeautifulSoup
import requests
import numpy as np

cities = ['matsue']

city_df = pd.read_csv('city_data.csv', index_col=0)

for city in cities:
  try:
    prec_no = city_df.at[city, 'prec_no']
    block_no = city_df.at[city, 'block_no']
    filename = f'data/{city}.csv'

    dataframe = pd.read_csv(filename)
    year = dataframe.iat[-1, 0]
    month = dataframe.iat[-1, 1]
    day = dataframe.iat[-1, 2] + 1

    td = datetime.timedelta(days=1)
    date = datetime.date(year, month, day)
    today = datetime.date.today()
    dl_days = (today - date).days

    for i in range(1, dl_days+1):
      progress = int(round(i / dl_days / 4 * 100))
      pro_bar = ('=' * progress) + (' ' * (25-progress))
      print('\rDownloading table of {0} in {1}  [{2}] {3}%'.format(str(date), city, pro_bar, round(i / dl_days * 100, 1)), end='')

      y = date.year
      m = date.month
      d = date.day

      url = 'https://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=' + str(prec_no) + '&block_no=' + str(block_no) + '&year=' + str(y) + '&month=' + str(m) + '&day=' + str(d) + '&view='
      headers = {
        'User-Agent': 'Contact: eku1gyps0phila@gmail.com'
      }

      res = requests.get(url)
      soup = BeautifulSoup(res.content, 'html.parser')

      table = soup.find(id='tablefix1')

      data = np.array([cell.get_text() for cell in table.find_all('td', class_='data_0_0')])
      data = data.reshape(-1, 16)[:, :11]

      date_cols = np.array([[y] * 24, [m] * 24, [d] * 24]).T
      hour_col = np.array(range(1, 25)).reshape(24, 1)
      data = np.concatenate([date_cols, hour_col, data], axis=1)
      df = pd.DataFrame(data, columns=['Year', 'Month', 'Day', 'Hour', 'Atm1', 'Atm2', 'Rain', 'Temp', 'DP', 'VP', 'Humidity', 'Wind str', 'Wind dir', 'Sun Shine', 'Insolation'])
      nan_cols = ['Atm1', 'Temp', 'Atm2', 'DP', 'VP', 'Humidity', 'Wind str']
      zero_cols = ['Rain', 'Sun Shine', 'Insolation']

      df[nan_cols] = df[nan_cols].mask(df[nan_cols].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull(), np.nan)
      df[zero_cols] = df[zero_cols].mask(df[zero_cols].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull(), 0.0)

      df = df.astype({
        'Year': 'int64',
        'Month': 'int64',
        'Day': 'int64',
        'Hour': 'int64',
        'Atm1': 'float64',
        'Atm2': 'float64',
        'Rain': 'float64',
        'Temp': 'float64',
        'DP': 'float64',
        'VP': 'float64',
        'Humidity': 'float64',
        'Wind str': 'float64',
        'Sun Shine': 'float64',
        'Insolation': 'float64',
      })

      direction = ['東', '東北東', '北東', '北北東', '北', '北北西', '北西', '西北西', '西', '西南西', '南西', '南南西', '南', '南南東', '南東', '東南東']
      east_map = {}
      north_map = {}
      for i in range(16):
        east_map[direction[i]] = np.cos(i/8*np.pi)
        north_map[direction[i]] = np.sin(i/8*np.pi)
      east_map['静穏'] = 0
      north_map['静穏'] = 0

      df['Wind E dir'] = df['Wind dir'].replace(east_map)
      df['Wind N dir'] = df['Wind dir'].replace(north_map)
      df[['Wind E dir', 'Wind N dir']] = df[['Wind E dir', 'Wind N dir']].mask(df[['Wind E dir', 'Wind N dir']].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull(), np.nan)

      df = df.astype({
        'Wind E dir': 'float64',
        'Wind N dir': 'float64',
      })

      df[['Wind E dir', 'Wind N dir']] = df[['Wind E dir', 'Wind N dir']].interpolate(limit_direction='both')
      df['Wind E'] = df['Wind str'] * df['Wind E dir']
      df['Wind N'] = df['Wind str'] * df['Wind N dir']

      df = df.drop(['Wind str', 'Wind dir', 'Wind E dir', 'Wind N dir'], axis=1)

      df = df.astype({
        'Wind E': 'float64',
        'Wind N': 'float64',
      })

      df[['Wind E', 'Wind N']] = df[['Wind E', 'Wind N']].round(1)

      dataframe = pd.concat([dataframe, df])
      time.sleep(1)
      date += td
  except Exception as e:
    print(e)

  finally:
    dataframe = dataframe.interpolate()
    dataframe.to_csv(filename, index=False)