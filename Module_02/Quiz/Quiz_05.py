### ♧♣ Quiz 05

# Logistic Regression 종합실습

import numpy as np
import pandas as pd

# Machine Learning Library
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# accuracy measure
from sklearn import metrics

# 1. 'data/fish.csv'파일을 로드하여 df (DataFrame)에 저장하고 내용을 출력하시오.

df = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/fish.csv")
df

# 2. 컬럼 'Depth'의 제곱값을 구하여 df (DataFrame)의 컬럼에 추가하시오.

df['D2'] = df['Depth']**2
df

# 3. 'Kg'과 'Depth'의 비율 (Kg/Depth)을 구하여 df (DataFrame)의 컬럼에 추가하시오.

df['DKgRatio'] = df['Kg'] / df['Depth']
df

# 4. Type이 'tuna' 인 경우에 1, 아닌 경우에 0으로 하는 컬럼 df (DataFrame)에 추가하시오.

df['isTuna'] = df['Type'].apply(lambda x: 1 if x == 'tuna' else 0)
df

# 5. Logistic Regression (tuna인 데이터와 아닌 데이터 분류)을 위한 모든 독립변수를 선택하시오.

col_list = ['Length','Depth']

# 6. 학습 데이터와 테스트 데이터를 분리(7:3)하시오.

X_train, X_test, y_train, y_test = train_test_split(
    df[col_list], df['isTuna'], test_size=0.3, random_state=123)

# 7. Logistic Regression 모델을 생성하고 학습하시오.

model = LogisticRegression()
model.fit(X_train, y_train)

# 8. 테스트 데이터를 이용하여 tuna인 데이터와 아닌 데이터를 예측하고 결과를 출력하시오.

prediction = model.predict(X_test)
prediction

# 9. 예측한 결과의 정확도 (Accuracy)를 구하시오.
# - metrics.accuracy_score() 함수를 이용하시오.

print('Accuracy - Logistic Regression:', metrics.accuracy_score(prediction, y_test))

# 10. 예측한 결과의 Confusion Matrix를 구하시오.
# - crosstab() 함수를 이용하시오.

pd.crosstab(prediction, y_test, margins=True)

# end
