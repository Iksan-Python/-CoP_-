# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:34:18 2021

@author: 찬채아빠
"""

# 301 Classification

## Fish Data

---

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline 

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 8
rcParams['font.family'] = 'New Gulim'
rcParams['font.size'] = 20
rcParams['axes.unicode_minus'] = False


# Machine Learning Library

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

# accuracy measure
from sklearn import metrics

#### 데이터 로드

df = pd.read_csv('data/fish.csv')
df

#### 길이의 제곱 컬럼 생성

df['L2'] = df['Length'] * df['Length']

#### 길이와 무게 비율 컬럼 생성

df['LKgRatio'] = df['Kg'] / df['Length']
df

#### 컬럼 추가(isTuna) - 참치: 1, 나머지: 0

df['isTuna'] = df['Type'].apply(lambda x: 1 if x == 'tuna' else 0)

#### 컬럼 추가( TypeNum) - 참치: 0, 연어: 1, 고등어:2

df['TypeNum'] = df['Type'].apply(lambda x: 0 if x == 'tuna' else 1 if x == 'salmon' else 2)
df

fig = plt.figure(figsize=(12,8))
plt.scatter(df['Length'], df['Kg'], c=df['TypeNum'], s=60)

plt.xlabel('Length')
plt.ylabel('Kg')

plt.show()

# 1. Logistic Regression

#col_list = ['Length','Depth','Kg','L2','LKgRatio']
col_list = ['Length','Depth']

# 데이터 분리: 학습 데이터 + 테스트 데이터
X_train, X_test, y_train, y_test = train_test_split(
    df[col_list], df['isTuna'], random_state=123)

# 모델 생성
model = LogisticRegression()

# 모델 학습
#model.fit(X_train, y_train.values.ravel())
model.fit(X_train, y_train)

# 결과 예측
prediction1 = model.predict(X_test)
prediction1

# 정확도 확인
print('Accuracy - Logistic Regression:', metrics.accuracy_score(prediction1, y_test))

# Score - precision, recall, f1-score
print(metrics.classification_report(y_test, prediction1))

# Confusion Matrix
pd.crosstab(prediction1, y_test, margins=True)

---

# 2. Support Vector Machine

#col_list = ['Length','Depth','Kg','L2','LKgRatio']
col_list = ['Length','Depth']

# 데이터 분리: 학습 데이터 + 테스트 데이터
X_train, X_test, y_train, y_test = train_test_split(df[col_list], df['TypeNum'], random_state=123)

# 모델 생성
model = SVC(kernel='linear', C=0.1, gamma=0.1)

# 모델 학습
#model.fit(X_train, y_train.values.ravel())
model.fit(X_train, y_train)

# 결과 예측
prediction2 = model.predict(X_test)
prediction2

# 정확도 확인
print('Accuracy - SVM:', metrics.accuracy_score(prediction2, y_test))

# Score - precision, recall, f1-score
print(metrics.classification_report(y_test, prediction2))

# Confusion Matrix
pd.crosstab(prediction2, y_test, margins=True)

## 2.1 Support Vector Machine - PCA

col_list = ['Length','Depth','Kg','L2','LKgRatio']

# PCA 모델 생성
pca = PCA(n_components=2)

# PCA Transform
df_pca = pca.fit_transform(df[col_list])
df_pca

# 데이터 분리: 학습 데이터 + 테스트 데이터
X_train, X_test, y_train, y_test = train_test_split(df_pca, df['TypeNum'], random_state=123)

# 모델 생성
model = SVC(kernel='linear', C=0.1, gamma=0.1)

# 모델 학습
#model.fit(X_train, y_train.values.ravel())
model.fit(X_train, y_train)

# 결과 예측
prediction2 = model.predict(X_test)
prediction2

# 정확도 확인
print('Accuracy - SVM:', metrics.accuracy_score(prediction2, y_test))

# Score - precision, recall, f1-score
print(metrics.classification_report(y_test, prediction2))

# Confusion Matrix
pd.crosstab(prediction2, y_test, margins=True)

---

# 3. Decision Tree

col_list = ['Length','Depth','Kg','L2','LKgRatio']

# 데이터 분리: 학습 데이터 + 테스트 데이터
X_train, X_test, y_train, y_test = train_test_split(df[col_list], df['TypeNum'], random_state=123)

# 모델 생성
model = DecisionTreeClassifier()

# 모델 학습
model.fit(X_train, y_train)

# 결과 예측
prediction3 = model.predict(X_test)
prediction3

# 정확도 확인
print('Accuracy - Decision Tree:', metrics.accuracy_score(prediction3, y_test))

# Score - precision, recall, f1-score
print(metrics.classification_report(y_test, prediction3))

# Confusion Matrix
pd.crosstab(prediction3, y_test, margins=True)

---

# 4. Random Forest

col_list = ['Length','Depth','Kg','L2','LKgRatio']

# 데이터 분리: 학습 데이터 + 테스트 데이터
X_train, X_test, y_train, y_test = train_test_split(df[col_list], df['TypeNum'], random_state=123)

# 모델 생성
model = RandomForestClassifier(n_estimators=340)

# 모델 학습
#model.fit(X_train, y_train.values.ravel())
model.fit(X_train, y_train)

# 결과 예측
prediction4 = model.predict(X_test)
prediction4

# 정확도 확인
print('Accuracy - Random Forests:', metrics.accuracy_score(prediction4, y_test))

# Score - precision, recall, f1-score
print(metrics.classification_report(y_test, prediction4))

# Confusion Matrix
pd.crosstab(prediction4, y_test, margins=True)

---

# end of file

# 302 Clustering & PCA

---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import rcParams
rcParams['font.family'] = 'New Gulim'
rcParams['font.size'] = 20
rcParams['axes.unicode_minus'] = False

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import DBSCAN

from sklearn.metrics.cluster import silhouette_score

from sklearn.decomposition import PCA

# 1. 보성군 날씨

#### 데이터 로드

# 보성군 날씨
df = pd.read_csv('data/bosung_weather.csv', encoding='cp949', parse_dates=['시간'])
df.set_index('시간', inplace=True)
df

# 2개 속성 선택, 결측치 제거, 200개 샘플링
df = df[['기온(°C)', '풍속(m/s)']].dropna().sample(n=200)
df

### 1.1 K-Means Clustering

# 모델 생성
km = KMeans(n_clusters=4, random_state=123)

# 군집 분류
labels_km = km.fit_predict(df)

labels_km

print("군집의 크기: {}".format(np.bincount(labels_km)))

# 2차원 시각화
fig = plt.figure(figsize=(12,8))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels_km, s=100, edgecolors='black')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=200,
            marker='^', c=range(km.n_clusters), linewidth=2, edgecolors='black')

plt.xlabel("기온")
plt.ylabel("풍속")
plt.show()

### 1.2 병합 군집 ( agglomerative clustering )

# 모델 생성
agg = AgglomerativeClustering(n_clusters=4)

# 군집 분류
labels_agg = agg.fit_predict(df)

labels_agg

print("군집의 크기: {}".format(np.bincount(labels_agg)))

# 2차원 시각화
fig = plt.figure(figsize=(12,8))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels_agg, s=100, edgecolors='black')

plt.xlabel("기온")
plt.ylabel("풍속")

plt.show()

#### 1.2.1 Dendrogram

# scipy의 계층 군집 유사도가 들어 있는 연결배열 반환
linkage_array = ward(df)

# 클러스터 사이의 거리가 담겨있는 linkage_array로 덴드로그램 시각화
# p값을 통해 최종 leaf 깊이 설정
plt.figure(figsize=(10,10))
dendrogram(linkage_array, p=3, truncate_mode='level', no_labels=True)

plt.xlabel("샘플 번호")
plt.ylabel("클러스터 거리")

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [20, 20], '--', c='k')

### 1.3 DBSCAN

# 모델 생성
dbscan = DBSCAN(min_samples=7, eps=1)

# 군집 분류
labels_dbscan = dbscan.fit_predict(df)

labels_dbscan

# 2차원 시각화
fig = plt.figure(figsize=(12,8))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels_dbscan, s=100, edgecolors='black')

plt.xlabel("기온")
plt.ylabel("풍속")

### 1.4 군집 분석 평가 - 실루엣 점수

fig, axes = plt.subplots(1, 3, figsize=(12, 3),subplot_kw={'xticks': (), 'yticks': ()})

rcParams['font.size'] = 15

axes[0].scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels_km, s=60, edgecolors='black')
axes[0].set_title("{} : {:.2f}".format(km.__class__.__name__, silhouette_score(df, labels_km)))

axes[1].scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels_agg, s=60, edgecolors='black')
axes[1].set_title("{} : {:.2f}".format(agg.__class__.__name__, silhouette_score(df, labels_agg)))

axes[2].scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels_dbscan, s=60, edgecolors='black')
axes[2].set_title("{} : {:.2f}".format(dbscan.__class__.__name__, silhouette_score(df, labels_dbscan)))

plt.show()

### 1.5 PCA - 주성분 분석

df = pd.read_csv('data/bosung_weather.csv', encoding='cp949', parse_dates=['시간'])
df.set_index('시간', inplace=True)
df

# 결측 제거 및 200개 샘플링
df = df.dropna().sample(n=200)

# PCA 모델 생성
pca = PCA(n_components=2)

# PCA Transform
df_pca = pca.fit_transform(df)
df_pca

# K-Means clustering
km = KMeans(n_clusters=4, random_state=123)
labels_km = km.fit_predict(df_pca)

# 병합 군집 (agglomerative clustering)
agg = AgglomerativeClustering(n_clusters=4)
labels_agg = agg.fit_predict(df_pca)

# DBSCAN
dbscan = DBSCAN(min_samples=7, eps=3)
labels_dbscan = dbscan.fit_predict(df_pca)

# 군집별 실루엣 점수 평가

fig, axes = plt.subplots(1, 3, figsize=(12, 3),subplot_kw={'xticks': (), 'yticks': ()})

rcParams['font.size'] = 15

axes[0].scatter(df_pca[:, 0], df_pca[:, 1], c=labels_km, s=60, edgecolors='black')
axes[0].set_title("{} : {:.2f}".format(km.__class__.__name__, silhouette_score(df, labels_km)))

axes[1].scatter(df_pca[:, 0], df_pca[:, 1], c=labels_agg, s=60, edgecolors='black')
axes[1].set_title("{} : {:.2f}".format(agg.__class__.__name__, silhouette_score(df, labels_agg)))

axes[2].scatter(df_pca[:, 0], df_pca[:, 1], c=labels_dbscan, s=60, edgecolors='black')
axes[2].set_title("{} : {:.2f}".format(dbscan.__class__.__name__, silhouette_score(df, labels_dbscan)))

plt.show()


---

# 2. 유럽 국가별 단백질 섭취원 비율

#### 데이터 로드

# 유럽국가 단백질 섭취원
df_p = pd.read_csv('data/protein.csv')
df_p.set_index('Country', inplace=True)
df_p

# 모델 생성
km_p = KMeans(n_clusters=5, random_state=123)

# 군집 분류
labels_km_p = km_p.fit_predict(df_p)

labels_km_p

df_p['gpnum'] = labels_km_p
df_p.sort_values(by='gpnum')

---

# end of file



# 303 참고: naver music 

import urllib
import requests
from bs4 import BeautifulSoup as bs
import re

# naver 뮤직의 Psy의 New Face 가사 주소 

#url = "http://music.naver.com/lyric/index.nhn?trackId=17740104"
# url = "http://music.naver.com/lyric/index.nhn?trackId=17822649"
url = "https://vibe.naver.com/track/17740104"

r = requests.get(url)

bs_rt = bs(r.text, "html.parser")
lyric = bs_rt.find("div", id="lyricText")
text  = re.sub(r"<.*?>", "\\n", str(lyric))

print (text)

psy_list = text.split()
psy_set = set(psy_list)
psy_set

url2 = "http://music.naver.com/listen/top100.nhn?domain=DOMESTIC"

r2 = requests.get(url2)

naver_music = bs(r2.text, "html.parser")

# Parsing
# 심볼의 문자열을 정규 문법을 따라 분석하는 프로세스

result = naver_music.find_all("a", title="가사")
title  = naver_music.find_all("span", class_="ellipsis")

title = naver_music.find_all("a", class_=re.compile("^_title"))

singer = naver_music.find_all("td", class_=re.compile("^_artist artist"))

singer_list = []

for idx, xx in enumerate(singer):
    if(idx == 0): continue
    
    print (idx, xx.text.strip())
    singer_list.append(xx.text.strip())


title_list = []

for idx, xx in enumerate(title):
    print (idx + 1, xx.text.strip())
    title_list.append(xx.text.strip())

p = re.compile(r"[0-9]{4,10}")
all_music = p.findall(str(result))

all_music

len(all_music)

# 이 아래 소스는 네이버 뮤직에서 가사를 가져오는 소스이다. 
# 최신 인기가요 50곡의 가사를 출력한다. 
# 매일 매일 순위가 변경된다.


url_pre = "http://music.naver.com/lyric/index.nhn?trackId="

for idx, xxx in enumerate(all_music):
    url = "{}{}".format(url_pre, xxx)
    #print (url)
    r = requests.get(url)

    bs_rt = bs(r.text, "html.parser")
    lyric = bs_rt.find("div", id="lyricText")
    text = re.sub(r"<.*?>", "\\n", str(lyric))
    print (title_list[idx])
    print (text)
    try:
        f = open('data/' + '{:02d}_'.format(idx+1) + title_list[idx] + '.txt', 'w', encoding='utf-8')
        f.write(text)
        f.close()
    except:
        pass



---

# 네이버에 자동 로그인하는 코드 전문은 아래와 같다.

# coding=utf-8
 
from selenium import webdriver
 
browser = webdriver.Chrome('/Users/open/Documents/chromedriver/chromedriver')
 
browser.get('http://www.naver.com')
 
login_bt=browser.find_element_by_class_name('lg_local_btn')
login_bt.click()
 
# ID를 입력한다
id=browser.find_element_by_id('id')
id.send_keys('ID입력')
 
# PWD를 입력 한다
id=browser.find_element_by_id('pw')
id.send_keys('PW입력')
 
naver_submit=browser.find_element_by_class_name('btn_global')
naver_submit.click()


# end of file



