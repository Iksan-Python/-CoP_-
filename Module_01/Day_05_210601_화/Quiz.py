### ♧♣ Quiz 01

# 1. 1 이상, 100 이하의 수 중에서 4의 배수 혹은 9의 배수의 총합을 구하시오.


# 2. 구구단의 7단과 8단을 출력하시오.
 

# 3. 다음의 문자를 모스부호로 암호화 하시오.

# 문자 : HE GETS UP LATE

# 모스부호
dic = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.':'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z'
}
 
# end




### ♧♣ Quiz 02

# 1. 다음 배열의 3의 배수 (0 포함)를 모두 300으로 치환하시오.



# 2. 다음 DataFrame의 각 행의 최대값과 평균의 차이, 각 열의 최대값과 평균의 차이를 각각 구하시오.
# - DataFrame의 apply() 함수를 이용하시오.


# 3. 다음 데이터 (ldata)를 아래와 같은 형태의 DataFrame으로 변환하시오.
# - 행: date
# - 열: item의 값
# - 값: value

data = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/macrodata.csv")

# end



### ♧♣ Quiz 03

# 1. 다음 tips 데이터의 day와 time별로 total_bill의 평균과 합계를 구하시오.
# - DataFrame의 groupby()함수를 이용하시오.

import pandas as pd

tips = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/tips.csv")
tips



# 2. 어느 device_id에 가장 많은 'WM_STATE' Log가 기록되어 있는지 Log수가 많은 순서대로 표시하시오.

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/washing_machine.csv")



# 3. 다음 데이터의 로그 생성 횟수를 create_dt_utc(4시간 단위)를 기준으로 bar chart를 그리시오.
# - Grouper(), groupby() 함수를 이용하시오

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/washing_machine.csv")
df

# end




### ♧♣ Quiz 04

# 1. 'data/titanic_train.csv'파일을 로드하여 df (DataFrame)에 저장하고 내용을 출력하시오.

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/heuiy/data/main/AI_Camp/titanic_train.csv")
df

# 2. df (DataFrame)의 컬럼 중 'Name', 'Fare', 'Cabin' 을 영구히 삭제하고 내용을 출력하시오.

# 3. df (DataFrame)의 성별 (Sex)에 따른 생존여부 (Survived)를 행렬(합계 포함)로 표시하시오.
# - crosstab() 함수를 이용하시오.

# 4. df (DataFrame)의 Age에 대한 결측치를 Sex(male, female), Pclass(1,2,3)로 구분하여 각 평균치로 채우시오.

# end

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

# 3. 'Kg'과 'Depth'의 비율 (Kg/Depth)을 구하여 df (DataFrame)의 컬럼에 추가하시오.

# 4. Type이 'tuna' 인 경우에 1, 아닌 경우에 0으로 하는 컬럼 df (DataFrame)에 추가하시오.

# 5. Logistic Regression (tuna인 데이터와 아닌 데이터 분류)을 위한 모든 독립변수를 선택하시오.

# 6. 학습 데이터와 테스트 데이터를 분리(7:3)하시오.

# 7. Logistic Regression 모델을 생성하고 학습하시오.

# 8. 테스트 데이터를 이용하여 tuna인 데이터와 아닌 데이터를 예측하고 결과를 출력하시오.

# 9. 예측한 결과의 정확도 (Accuracy)를 구하시오.
# - metrics.accuracy_score() 함수를 이용하시오.

# 10. 예측한 결과의 Confusion Matrix를 구하시오.
# - crosstab() 함수를 이용하시오.

# end
