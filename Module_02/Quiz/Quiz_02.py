
### ♧♣ Quiz 02

# 1. 다음 배열의 3의 배수 (0 포함)를 모두 300으로 치환하시오.

import numpy as np

data = np.arange(28)
data = data.reshape(7,4)

data

data = np.where(data % 3 == 0, 300, data)
data

# 2. 다음 DataFrame의 각 행의 최대값과 평균의 차이, 각 열의 최대값과 평균의 차이를 각각 구하시오.
# - DataFrame의 apply() 함수를 이용하시오.

import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(12).reshape(4,3),
                  columns=['col1', 'col2', 'col3'],
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])

np.abs(df)

f = lambda x: x.max() - x.mean()

df.apply(f, axis=1)

df.apply(f, axis=0)

# 3. 다음 데이터 (ldata)를 아래와 같은 형태의 DataFrame으로 변환하시오.
# - 행: date
# - 열: item의 값
# - 값: value

data = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/macrodata.csv")

periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
columns = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
data = data.reindex(columns=columns)
data.index = periods.to_timestamp('D', 'end')

ldata = data.stack().reset_index().rename(columns={0: 'value'})

ldata.head()

pivoted = ldata.pivot('date', 'item', 'value')
pivoted.head()

# end
