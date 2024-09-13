import pandas as pd

def load(file_path):
  df = pd.read_csv(file_path)
  m, s = df['tow'].mean(), df['tow'].std()
  df['date'] = pd.to_datetime(df['date'])
  df['actual_offblock_time'] = pd.to_datetime(df['actual_offblock_time'])
  df['arrival_time'] = pd.to_datetime(df['arrival_time'])
  df['offblock_hour'] = df['actual_offblock_time'].dt.round('h').dt.hour
  df['offblock_day_of_week'] = df['actual_offblock_time'].dt.dayofweek
  df['offblock_month'] = df['actual_offblock_time'].dt.month
  df = df.drop(columns=['flight_id','name_adep', 'callsign', 'actual_offblock_time', 'arrival_time', 'date'])
  df['tow'] = (df['tow']-m)/s
  return df, m, s # Roubando aqui na média e desvio padrão do dataset inteiro :p

