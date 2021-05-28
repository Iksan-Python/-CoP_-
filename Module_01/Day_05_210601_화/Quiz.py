
df1 = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/titanic/gender_submission.csv")

df2 = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/titanic/train.csv")

df3 = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/titanic/test.csv")

df4 = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/weather/weather.csv")

df = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AICamp/titanic.csv")

### ♧♣ Homework 01

# # python homework set1 01

# ### 1. 1 이상, 100 이하의 수 중에서 4의 배수 혹은 9의 배수의 총합을 구하시오.

total = 0

for n in range(1, 101):
    if n % 4 == 0 or n % 9 == 0 :
        total += n
print(total)


# ### 2. 구구단의 7단과 8단을 출력하시오.

num = 0
for i in range(1, 10):
    print('7 *', i, '=', 7*i)
for i in range(1, 10):
    print('8 *', i, '=', 8*i)
   

# ### 3. 다음의 문자를 모스부호로 암호화 하시오.
# #### HE GETS UP LATE

# 모스부호
dic = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.':'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z'
}
 
dic_r = { v:k for k,v in dic.items()}
dic_r

def m_encode( alph ):
    alph = alph.upper()
    code = ''
    for word in alph.split():
        for char in word:
            code += dic_r.get(char, 'X') + ' '
        code += ' '
    return code.strip()

alpha = 'HE GETS UP LATE'

res = m_encode(alph)
print(res)


# end of file



### ♧♣ Homework 02
#!/usr/bin/env python
# coding: utf-8

# # python homework set1 02

# ### 1. 다음 배열의 3의 배수 (0 포함)를 모두 300으로 치환하시오.

import numpy as np

data = np.arange(28)
data = data.reshape(7,4)

data

data = np.where(data % 3 == 0, 300, data)
data


# ### 2. 다음 DataFrame의 각 행의 최대값과 평균의 차이, 각 열의 최대값과 평균의 차이를 각각 구하시오.
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

# ### 3. 다음 데이터 (ldata)를 아래와 같은 형태의 DataFrame으로 변환하시오.
# - 행: date
# - 열: item의 값
# - 값: value

data = pd.read_csv('data/macrodata.csv')

periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
columns = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
data = data.reindex(columns=columns)
data.index = periods.to_timestamp('D', 'end')

ldata = data.stack().reset_index().rename(columns={0: 'value'})

ldata.head()

pivoted = ldata.pivot('date', 'item', 'value')
pivoted.head()


# end of file

### ♧♣ Homework 03

# # python homework set1 03

# ### 1. 다음 tips 데이터의 day와 time별로 total_bill의 평균과 합계를 구하시오.
# - DataFrame의 groupby()함수를 이용하시오.

import pandas as pd

tips = pd.read_csv('data/tips.csv')
tips

tips = tips.drop(['tip', 'size'], axis=1)

grouped = tips.groupby(['day', 'time'])
grouped.agg(['mean', 'sum'])

# ### 2. 어느 device_id에 가장 많은 'WM_STATE' Log가 기록되어 있는지 Log수가 많은 순서대로 표시하시오.

import pandas as pd


df = pd.read_csv('data/washing_machine.csv')

df[df['event_type'] == 'WM_STATE']['device_id'].value_counts()


# ### 3. 다음 데이터의 로그 생성 횟수를 create_dt_utc(4시간 단위)를 기준으로 bar chart를 그리시오.
# - Grouper(), groupby() 함수를 이용하시오

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('data/washing_machine.csv')
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


# end of file



### ♧♣ Homework 04

# # python homework set1 04

# ### 1. 'data/titanic_train.csv'파일을 로드하여 df (DataFrame)에 저장하고 내용을 출력하시오.

import pandas as pd

df = pd.read_csv('data/titanic_train.csv')
df

# ### 2. df (DataFrame)의 컬럼 중 'Name', 'Fare', 'Cabin' 을 영구히 삭제하고 내용을 출력하시오.

df = df.drop(['Name', 'Fare', 'Cabin'],
            axis=1)
df

# ### 3. df (DataFrame)의 성별 (Sex)에 따른 생존여부 (Survived)를 행렬(합계 포함)로 표시하시오.
# - crosstab() 함수를 이용하시오.

pd.crosstab(df.Sex, df.Survived, margins=True)

# ### 4. df (DataFrame)의 Age에 대한 결측치를 Sex(male, female), Pclass(1,2,3)로 구분하여 각 평균치로 채우시오.

mean_m_1 = df[(df['Sex']=='male')   & (df['Pclass']==1)]['Age'].mean()
mean_m_2 = df[(df['Sex']=='male')   & (df['Pclass']==2)]['Age'].mean()
mean_m_3 = df[(df['Sex']=='male')   & (df['Pclass']==3)]['Age'].mean()
mean_f_1 = df[(df['Sex']=='female') & (df['Pclass']==1)]['Age'].mean()
mean_f_2 = df[(df['Sex']=='female') & (df['Pclass']==2)]['Age'].mean()
mean_f_3 = df[(df['Sex']=='female') & (df['Pclass']==3)]['Age'].mean()

df['Age'].isnull().sum()

df.loc[(df['Age'].isnull())&(df['Sex']=='male')  &(df['Pclass']==1),'Age'] = mean_m_1
df.loc[(df['Age'].isnull())&(df['Sex']=='male')  &(df['Pclass']==2),'Age'] = mean_m_2
df.loc[(df['Age'].isnull())&(df['Sex']=='male')  &(df['Pclass']==3),'Age'] = mean_m_3
df.loc[(df['Age'].isnull())&(df['Sex']=='female')&(df['Pclass']==1),'Age'] = mean_f_1
df.loc[(df['Age'].isnull())&(df['Sex']=='female')&(df['Pclass']==2),'Age'] = mean_f_2
df.loc[(df['Age'].isnull())&(df['Sex']=='female')&(df['Pclass']==3),'Age'] = mean_f_3

df['Age'].isnull().sum()

# end of file



### ♧♣ Homework 05

# # python homework set1 05

# ### Logistic Regression 종합실습

import numpy as np
import pandas as pd

# Machine Learning Library
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# accuracy measure
from sklearn import metrics


# ### 1. 'data/fish.csv'파일을 로드하여 df (DataFrame)에 저장하고 내용을 출력하시오.

df = pd.read_csv('data/fish.csv')
df


# ### 2. 컬럼 'Depth'의 제곱값을 구하여 df (DataFrame)의 컬럼에 추가하시오.

df['D2'] = df['Depth']**2
df

# ### 3. 'Kg'과 'Depth'의 비율 (Kg/Depth)을 구하여 df (DataFrame)의 컬럼에 추가하시오.

df['DKgRatio'] = df['Kg'] / df['Depth']
df

# ### 4. Type이 'tuna' 인 경우에 1, 아닌 경우에 0으로 하는 컬럼 df (DataFrame)에 추가하시오.

df['isTuna'] = df['Type'].apply(lambda x: 1 if x == 'tuna' else 0)
df

# ### 5. Logistic Regression (tuna인 데이터와 아닌 데이터 분류)을 위한 모든 독립변수를 선택하시오.

col_list = ['Length','Depth']

# ### 6. 학습 데이터와 테스트 데이터를 분리(7:3)하시오.

X_train, X_test, y_train, y_test = train_test_split(
    df[col_list], df['isTuna'], test_size=0.3, random_state=123)

# ### 7. Logistic Regression 모델을 생성하고 학습하시오.

model = LogisticRegression()
model.fit(X_train, y_train)

# ### 8. 테스트 데이터를 이용하여 tuna인 데이터와 아닌 데이터를 예측하고 결과를 출력하시오.

prediction = model.predict(X_test)
prediction

# ### 9. 예측한 결과의 정확도 (Accuracy)를 구하시오.
# - metrics.accuracy_score() 함수를 이용하시오.

print('Accuracy - Logistic Regression:', metrics.accuracy_score(prediction, y_test))

# ### 10. 예측한 결과의 Confusion Matrix를 구하시오.
# - crosstab() 함수를 이용하시오.

pd.crosstab(prediction, y_test, margins=True)

# end of file
