### ♧♣ Quiz 03

# 1. 다음 tips 데이터의 day와 time별로 total_bill의 평균과 합계를 구하시오.
# - DataFrame의 groupby()함수를 이용하시오.

import pandas as pd

tips = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/tips.csv")
tips

tips = tips.drop(['tip', 'size'], axis=1)

grouped = tips.groupby(['day', 'time'])
grouped.agg(['mean', 'sum'])

# 2. 어느 device_id에 가장 많은 'WM_STATE' Log가 기록되어 있는지 Log수가 많은 순서대로 표시하시오.

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/washing_machine.csv")

df[df['event_type'] == 'WM_STATE']['device_id'].value_counts()

# 3. 다음 데이터의 로그 생성 횟수를 create_dt_utc(4시간 단위)를 기준으로 bar chart를 그리시오.
# - Grouper(), groupby() 함수를 이용하시오

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/washing_machine.csv")
df

df['create_dt_utc'].head

df['create_dt_utc'][0]

type(df['create_dt_utc'][0])

df['create_dt_utc'] = pd.to_datetime(df['create_dt_utc'])

type(df['create_dt_utc'][0])

grouper = pd.Grouper(key='create_dt_utc', freq='4h')
grouper

gp_4h = df.groupby(grouper)

gp_4h.count().head()

freq = gp_4h['create_dt_utc'].count()
freq

freq.plot(kind='bar', figsize=(15, 7))

freq.tz_localize('UTC').tz_convert('Asia/Seoul').plot(kind='barh', figsize=(15, 15))

# end
