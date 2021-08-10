### ♧♣ Quiz 04

# 1. 'data/titanic_train.csv'파일을 로드하여 df (DataFrame)에 저장하고 내용을 출력하시오.

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/titanic_train.csv")
df

# 2. df (DataFrame)의 컬럼 중 'Name', 'Fare', 'Cabin' 을 영구히 삭제하고 내용을 출력하시오.

df = df.drop(['Name', 'Fare', 'Cabin'],
            axis=1)
df

# 3. df (DataFrame)의 성별 (Sex)에 따른 생존여부 (Survived)를 행렬(합계 포함)로 표시하시오.
# - crosstab() 함수를 이용하시오.

pd.crosstab(df.Sex, df.Survived, margins=True)

# 4. df (DataFrame)의 Age에 대한 결측치를 Sex(male, female), Pclass(1,2,3)로 구분하여 각 평균치로 채우시오.

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

# end
