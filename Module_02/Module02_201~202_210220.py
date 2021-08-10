# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:31:15 2021

@author: 찬채아빠
"""

# Data 분석을 위한 패키지

1. NumPy
    - Numerical Python
    - 과학계산용 패키지
    - 특징
        - 빠르고 효율적인 다차원 배열 객체 ndarray (n-dimensional array)
            - `[[1, 2], [1, 2], [[1, 2], [3, 4]]]`
        - 데이터 분석 시 데이터 컨테이너 역할(데이터를 담는 그릇)
    - 설치
        - `pip install numpy`
1. pandas
    - 금융회사에 다니고 있던 Wes McKinney가 처음에 금융 데이터 분석을 위해 설계(2008년)
    - pandas: 계량 경제학 용어인 Panel Data와 Analysis의 합성어
    - Panel Data<sup>1</sup>
        - multi-dimensional data involving measurements over time (Wikipedia)
    - 구조화된 데이터를 빠르고 쉬우면서 다양한 형식으로 가공할 수 있는 풍부한 자료 구조와 함수를 제공한다.
    - pandas의 기본 구조는 NumPy로 되어있다.
    - 설치
        - `pip install pandas`
1. matplotlib/seaborn/bokeh
    - 시각화 도구
    - 설치
        - `pip install matplotlib`
        - `pip install seaborn`
        - `pip install bokeh`
        
1. SciPy
    - 과학 계산용 패키지
    - 최적화, 선형대수, 적분 등 과학 계산에 쓰이는 대부분이 있음
    - Numpy extension of Python

1. scikit-learn
    - SciPy에 기반한 머신러닝 프레임워크

<img src="https://www.altexsoft.com/media/2017/08/%5E4275D3AF463329B3C66C6157C233DBF8F6B2BCCF49CC64ABE9%5Epimgpsh_fullsize_distr.png">

---

# NumPy: 배열과 벡터 계산

### NumPy Documentation

[Tutorials](https://docs.scipy.org/doc/numpy/user/quickstart.html)

#### Numerical Python의 줄임말

1. 빠르고 효율적인 메모리 사용, 벡터 연산, 브로드캐스팅(확대) 기능을 제공하는 다차원 배열인 ndarray (n dimentional array)를 제공
1. for 문 등 반복문을 작성할 필요없이 전체 배열에 대해 빠른 연산을 제공
1. 배열 데이터를 디스크에 쓰거나 읽을 수 있는 도구
1. 선형대수, 난수 발생기, 푸리에(Fourier) 변환 기능
1. C, C++, 포트란 등 다른 언어로 쓰여진 코드를 통합하는 도구

#### 데이터 분석에서 빠른 연산을 위해 자주 사용하는 기능

1. 배열에서 데이터 변경, 정제, 부분 집합, 필터링의 빠른 수행
2. 정렬, 유일 원소 찾기, 집합 연산
3. 통계 표현과 데이터의 수집/요약
4. 여러 데이터의 병합, 데이터 정렬과 데이터 조작

#### 표준 NumPy의 컨벤션을 import numpy as np로 사용

---

# 1. NumPy ndarray: 다차원 배열 객체

### 1.1 ndarray 사용

import numpy as np 

lst = [1, -2, -3, 4,  5,  6]

Array :

- (인상적인) 집합체[모음/무리]

- (메모리) 배열

- 파이썬에서는 [리스트] 로 묶인 애들을 다룰 때 Array 가 활용됨(hey).

- array 는 항상 ([리스트]) 를 안에 가지고 있음(hey).

arr = np.array(lst)
arr

arr * 3

arr + arr

#### <참고> 파이썬 리스트의 산술 연산

lst * 3

lst + lst

### 1.2 ndarray 생성

# 1차원 배열
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1

# 2차원 배열
data2 = [[1,2,3,4], [5,6,7,8]]
arr2 = np.array(data2)
arr2

arr1.ndim

# 1차원

arr2.ndim

# 2차원

arr1.shape

# R에서 dim() 과 비슷함
# 1차원, 5개

arr2.shape

# 2차원, 4개

arr2.reshape((4,2))

# 4차원, 2개로 변경

arr2.ravel()

# ravel : 단순히 전부 나열하라는 의미인 듯

arr2.shape

# ravel 해도 arr2 는 기존과 동일함

arr2.flatten()    # 항상 복사본 생성

# 차원 구분 없이 flat 하게 만들라는 의미인 듯

flatten :
- 납작[반반]해지다, 납작하게[반반하게] 만들다
- (건물·나무 등을) 깨부수다[넘어뜨리다]
- (시합 등에서 상대방의) 코를 납작하게 만들다 (=smash, thrash)

#### <참고> np.array는 생성될 때 적절한 자료형을 선택한다.

arr1.dtype

# 데이터 타입인듯

arr2.dtype

#### np.zeros(): 0 으로 초기화된 배열 생성

np.zeros(10) 

np.zeros((3,6)) ## 3행 6열

np.ones((4,7))

#### np.empty(): 초기화 되지 않은 배열 생성

np.empty((2,5)) 

#### np.arange(): 파이썬 range() 함수의 배열 버전

np.arange(15)

함수| 설명
:---| :---
array | 입력 데이터(리스트, 튜플, 배열 또는 다른 순차형 데이터)를 ndarray로 변환하며 dtype이 명시되지 않은 경우에는 자료형을 추론하여 저장한다. 기본적으로 입력 데이터는 복사된다.
asarray | 입력 데이터를 ndarray로 변환하지만 입력 데이터가 이미 ndarray일 경우, 복사가 되지 않는다.
arange | 내장 range 함수와 유사하지만 리스트 대신 ndarray를 반환한다.
ones, ones_like | 주어진 dtype과 주어진 모양을 가지는 배열을 생성하고 내용을 모두 1로 초기화한다. ones_like는 주어진 배열과 동일한 모양과 dtype을 가지는 배열을 새로 생성하여 내용을 모두 1로 초기화한다.
zeros, zeros_like | ones, ones_like와 같지만 내용을 0으로 채운다.
empty, empty_like | 메모리를 할당하여 새로운 배열을 생성하지만 ones나 zeros처럼 값을 초기화하지는 않는다.
eye, identity | N x N 크기의 단위 행렬을 생성한다(좌상단에서 우하단을 잇는 대각선은 1로 채워지고 나머지는 0으로 채워진다).

### 1.3 ndarray의 자료형

dtype로 자료형을 알 수 있다.

- 산술 데이터의 dtype은 float, int 같은 자료형의 이름과 하나의 원소가 차지하는 비트로 이루어짐
- 파이썬의 float 객체에서 사용되는 부동소수점 값은 8바이트 혹은 64바이트로 이루어짐
- [NumPy 자료형](http://docs.scipy.org/doc/numpy/user/basics.types.html)

arr1 = np.array([1,2,3], dtype=np.float32)
arr1

arr2 = np.array([1,2,3], dtype=np.float64)
arr2

arr1.dtype, arr2.dtype

#### 타입 변경:  astype() 메소드

arr = np.array([1,2,3,4,5])
arr

arr.dtype

f_arr = arr.astype(np.float64)
f_arr

f_arr.dtype

arr.dtype

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr

arr.astype(np.int32)

num_str = np.array(['1.25', '-9.6', '42'], dtype=np.string_)

num_str

num_str.astype(float) 

#### <참고> dtype 축약코드

종류 | Type Code | 설명
:---|:---|:---
int8,  uint8  | i1, u1   | 부호가 있는 8비트(1바이트) 정수형과 부호가 없는 8비트 정수형
int16, uint16 | i2, u2   | 부호가 있는 16비트 정수형과 부호가 없는 16비트 정수형
int32, uint32 | i4, u4   | 부호가 있는 32비트 정수형과 부호가 없는 32비트 정수형
int64, uint64 | i8, u8   | 부호가 있는 64비트 정수형과 부호가 없는 64비트 정수형
float16       | f2       | 반정밀도 부동소수점
float32       | f4 or f  | 단정밀도 부동소수점 C 언어의 float 형과 호환
float64       | f8 or d  | 배정밀도 부동소수점 C 언어의 double 형과 파이썬의 float 객체와 호환
float128      | f16 or g | 확장 정밀도 부동소수점
bool          | ?        | True, False 값을 저장하는 불리언형
object        | O        | 파이썬 객체형
string_       | S        | 고정 길이 문자열형, 길이가 10인 문자열의 dtype은 S10
unicode_      | U        | 고정 길이 유니코드형(예: U10)


emp_uint32 = np.empty(8, dtype='u4')
emp_uint32

### 1.4 배열과 스칼라 간의 연산

- **배열은 for 반복문을 작성하지 않고 데이터를 일괄처리할 수 있다(벡터화)**
- 같은 크기의 배열 간 산술연산은 배열의 각 요소 단위로 적용

arr = np.array([[1., 2., 3.], [4., 5., 6.]])

arr * 3

arr * arr

arr - arr

1 / arr

# **는 square root of x
arr ** 0.5

### 1.5 인덱싱(색인)과 슬라이싱

arr = np.arange(10)
arr

arr[5]

arr[5:8] 

arr[5:8] = 12

# array 5, 6, 7 대신 12, 12, 12 로 바꾸라는 의미

arr

#### 브로드캐스팅
- arr[5:8] = 배열에 스칼라 값을 대입하면 이 값이 필요한 만큼 복사되어 확산 전파
- 파이썬 리스트는 브로드캐스팅 불가
- 브로드캐스팅을 통해 2 x 2 array 와 1 x 2 array 가 사칙연산이 가능하도록 해줌

#### 배열 조각은 원본 배열의 View를 리턴 (파이썬 리스트의 슬라이싱은 복사본을 리턴)

- 즉, 데이터는 복사되지 않으며 뷰에 대한 변경은 그대로 원본 배열에 반영
- 복사본이 필요할 경우 arr[5:8].copy()를 사용

#### Slicing : 사용 예

- [:]: 배열 전체
- [0:n]: 0번째부터 n-1번째까지, 즉 n번 항목은 포함하지 않는다.
- [:5]: 0번째부터 4번째까지,5번은 포함하지 않는다.
- [2:]: 2번째부터 끝까지
- [-1]: 제일 끝에 있는 배열값 반환
- [-2]: 제일 끝에서 두번째 값 반환

arr = np.arange(10)
arr

arr_cp = arr[5:8].copy()

arr_cp

arr_cp[:] = 128
arr_cp

arr

#### 2차원, 3차원 배열 제어

arr2d = np.array([[1,2,3], [4,5,6], [7,8,9]])

arr2d.ndim

arr2d.shape

# 3차원, 3개

arr2d[2]

arr2d[0][2]

# ,로도 구분 가능
arr2d[0, 2]

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d

arr3d.ndim

arr3d[0]

arr_org = arr3d[0].copy()
arr_org

arr3d[0] = 42

arr3d

arr3d[0] = arr_org

arr3d

arr3d[1, 0]

#### 슬라이스 색인

arr = np.arange(10)
arr

arr[1:6]

arr2d = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr2d

arr2d[:2]

#### arr2d[:2, 1:]
     행,  열

- 같은 [ ] 안에서는 행과 열 구분

arr2d[:2, 1:]

# 같은 [] 내에서 행과 열 구분하여 표기

arr2d[:2][1:]

# [:2] 먼저 추린 다음에 다시 [1:] 선택함

arr2d[1, :2]

arr2d[2, :1]

arr2d[:, :1]

arr2d[:2, 1:] = 0

arr2d

### 1.6 불리언 색인

import numpy as np

data = np.arange(28)
data

data=data.reshape(7,4)
data

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

names

names == 'Bob'

data[names == 'Bob']

# 위에서 정의한 data array 중에서 첫 번째, 네 번째 행만 추려줌

names2 = np.array(['Bob', 'Joe', 'Will'])

names2 == 'Bob'

# data[names2 == 'Bob']

# 오류가 나는 이유?
# 위에서 정의한 data array 는 dimension 이 7인데,
# names2 array 는 dimension 이 3 이라서 corresponding 하지 않음

data[names == 'Bob', 2:]

data[names == 'Bob', 3]

#### != ,  ~ 으로 부정

names != 'Bob'

data

names

data[~(names == 'Bob')]

# 1~7 중에서, 2, 3, 5, 6, 7 총 5개 dimension 을 선택함

mask = (names == 'Bob') | (names == 'Will')

mask

data[mask]

data[mask, 3:]

#### 불리언 색인시 항상 데이터 복사

- 파이썬 예약어인 and, or는 불리언 배열에서는 사용 불가

data[data < 5] = 0

data

data[names == 'Bob'] = 7

# 0과 3 Dimension 만 7로 변환

data

### 1.7 팬시 색인

arr = np.empty((8, 4))

for i in range(8):
    arr[i] = i

arr

# 특정한 순서로 로우 선택
arr[[4, 3, 0, 6]]

arr[[-2, -3]]

# 뒤에서 두번째 Dimension, 뒤에서 3번째 Dimension

np.arange(32)

arr = np.arange(32).reshape((8, 4))

arr

# 1개 항목을 선택
# (1, 0), (5, 3), (7, 1), (2, 2)에 대응하는 요소가 선택
arr[[1, 5, 7, 2], [0, 3, 1, 2]]

# 특정 행들을 선택하고 이의 열의 순서를 바꾸는 경우
# 1, 5, 7, 2 행을 선택
# 컬럼 열에서 :,로 모든 행을 선택해 주어야 한다.
# 0, 3, 1, 2 열을 순서대로 선택

arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]

# np.ix_ 사용하는 방법
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]

### 1.8 배열 전치와 축 바꾸기

arr = np.arange(15).reshape((3, 5))
arr

arr.T # 행과 열을 바꿔준다.

arr.transpose() # T와 같은 역할을 한다.

arr = np.random.randn(6, 3)
arr

arr.T

np.dot(arr.T, arr) # arr.T 3 * 6 행렬 곱하기 arr 6 * 3 행렬의 결과값이다.

---

# 2. 유니버설 함수

- ufunc 유니버설 함수(universal function)는 ndarray 안에 있는 데이터 원소별로 연산을 수행하는 함수
- 유니버설 함수는 하나 이상의 스칼라 값을 받아서 하나 이상의 스칼라 결과 값을 반환하는 간단한 함수를 고속으로 수행할 수 있는 벡터화된 래퍼 함수

### 2.1 단항 유니버설 함수

import numpy as np 

arr = np.arange(10)

arr

np.sqrt(arr)

np.exp(arr)

### 2.2 이항 유니버설 함수

x = np.random.randn(8)

# 실행할 때마다 x 가 바뀜

y = np.random.randn(8)

# 실행할 때마다 y 가 바뀜

x

y

np.maximum(x, y)

# x, y 중 큰걸 선택하게 됨

#### [Universal functions](http://docs.scipy.org/doc/numpy/reference/ufuncs.html)

---

# 3. 배열을 사용한 데이터 처리

- 반복문 작성하지 않고 간결한 배열연산을 통해 많은 종류의 데이터 처리 작업
- **벡터화:** 배열연산을 사용해서 반복문을 명시적으로 제거하는 기법
- 순수 파이썬보다 2~3배, 많게는 수십, 수백 배까지 빠르다.
- **브로드캐스팅:** 아주 강력한 벡터 연산 방법

### 3.1 벡터화

import numpy as np 

points = np.arange(100000, dtype='f')
points[:10]

%%time
# 아래 내용 계산에 소요되는 시간을 측정하라는 의미인 듯
# ms 미리 세컨드로 시간 측정함

# %time
# ns 나노 세컨드로 시간 측정함

total = 0
for i in points:
    total += i
print(total)

# points 가 0 부터 십만까지니까 0부터 십만까지 다 합하라는 의미인 듯

# sum(0:100001)
# 이런 코드 아님

points.sum()

%time points.sum()

### 3.2 배열연산으로 조건절 표현하기

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

# xarr, yarr, cond 를 위와 같이 설정함
# cond 에서 True 면 x 를 선택할지, y 를 선택할지의 조건은 아래에서 설정함

result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
result

# x if c else y 
# if c 에서 True 면 x, False 면 y 를 선택하라는 의미임

# zip(*iterable)은 동일한 개수로 이루어진 자료형을 묶어 주는 역할을 하는 함수
# iterable은 반복 가능(iterable)한 자료형 여러 개를 입력할 수 있다는 의미임

result = np.where(cond, xarr, yarr)
result

arr = np.random.randn(4, 4)
arr

np.where(arr > 0, 2, -2)

# numpy 에서 where 함수는 리스트에서 index 함수와 비슷함

#         내가 어디서 찾을지
# numpy.where(numpy배열 == 찾을 값)
#                         무슨 값을 찾을지

np.where(arr > 0, 2, arr)

# 결과값이 왜 이렇게 나오는지 이해 안됨

### 3.3 수학 메서드와 통계 메서드

arr = np.random.randn(5, 4)
arr

# 당연히 실행할 때마다 결과값이 변경됨

arr.mean()

np.mean(arr)

# 위 arr.mean() 와 결과는 동일함

arr.sum()

#### 연산을 진행할 축을 선택( 행방향: axis = 0, 열방향: axis = 1) - Default: axis = 0

arr.sum(axis=0)

arr.sum(0)

arr.sum(axis=1)

# 위에서 정의한 arr 는 행과 열로 이루어진 2차원임.
# 그래서 arr.sum(axis=2)는 정의되지 않았음. 3차원일 때 axis 2 가 가능함.

arr.sum(1)

arr.mean(axis=1)

#### 누적연산 - cumsum(), cumprod(): 중간 계산 값을 담고 있는 배열을 반환

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr

arr.cumsum(axis=0)

# accumulate 누적하다

arr.cumprod(axis=1)

# accumulate 누적하다
# cumprod 누적해서 곱하라

arr.cumprod(axis=0)

#### 기본 배열 통계 메서드

메세드 | 설명
:---|:---
sum            | 배열 전체 혹은 특정 축에 대한 모든 원소의 합을 계산한다.
mean           | 산술평균을 구한다. 크기가 0인 배열에 대한 mean 결과는 NaN이다.
std, var       | 각각 표준편차와 분산을 구한다.
min, max       | 최소값, 최대값
argmin, argmax | 최소 원소의 색인 값, 최대 원소의 색인 값
cumsum         | 각 원소의 누적 합
cumprod        | 각 원소의 누적 곱


### 3.4 불리언 배열을 위한 메서드

- 불리언 값은 True = 1, False = 0 으로 취급
- 불리언 배열에 대한 sum 메서드를 실행하면 True인 원소의 개수를 반환

arr = np.random.randn(100)

arr

(arr > 0).sum()

# 100개 중 0 이상인 애들이 몇 개인지를 물어보는 듯

arr[:10]

# 위에서부터 두 줄을 나열하라

bools = np.array([False, False, True, False])

# 하나라도 True 이면 결과는 True

bools.any()

# 모두 True 이면 결과는 True
bools.all()

bools = np.array([True, True, True, True])

bools.any()

# 불리언 값은 True = 1, False = 0 으로 취급

bools.all()

#### 배열의 값이 0 이면 False, 그 외 나머지는 모두 True로 취급

arr

arr.any()

arr.all()

### 3.5 정렬

arr = np.random.randn(8)
arr

np.sort(arr)

arr

# <참고> 원본을 변경하지 않음
# np.에서는 sort 한 뒤에도 해도 원본이 안 바뀜.
# 원본은 크기순대로 정렬되지 않은 원본 그대로의 값

arr.sort()
arr

#  arr.sort() 한 뒤에 다시 arr 출력하면 sort 된 값으로 원본이 변경되어 있음

#### 다차원 배열의 정렬

arr = np.random.randn(5, 3)
arr

# axis = 0: 세로방향 정렬(column sort)
arr.sort(axis=0)
arr

# axis = 1: 가로방향 정렬(row sort)
arr.sort(axis=1)
arr

### 3.6 집합 함수

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

np.unique(names)

# 반복되는 이름 빼고 진짜 애들만 나열

ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])

np.unique(ints)

#### np.in1d()

values = np.array([6, 0, 0, 3, 2, 5, 6])

np.in1d(values, 2)

np.in1d(values, [2, 3, 6])

#### 배열 집합 연산

Method | Comment
:---|:---
unique(x)         | 배열 x에서 중복된 원소를 제거한 후 정렬하여 반환한다.
intersect1d(x, y) | 배열 x와 y에 공통적으로 존재하는 원소를 정렬하여 반환한다.
union1d(x, y)     | 두 배열의 합집합을 반환한다.
in1d(x, y)        | x의 원소 중 y의 원소를 포함하는지를 나타내는 불리언 배열을 반환한다.
setdiff1d(x, y)   | x와 y의 차집합을 반환한다.
setxor1d(x, y)    | 한 배열에는 포함되지만 두 배열 모두에는 포함되지 않는 원소들의 집합인 대칭차집합을 반환한다.

---

# 4. 배열의 파일 입∙출력

### 4.1 배열을 바이너리 형식으로 디스크에 저장하기

arr1 = np.arange(10)

arr2 = np.arange(5)

np.save('data/file_arr1.npy', arr1)

# 주피터 노트북 실행되는 폴더에 data 라는 폴더를 만들어줘야 저장됨

# np.save 는 npy 가 생성됨
# np.savez 는 npz 가 생성됨

# 위에서 만든 npy 열어보면 이런 에러가 뜸
# 	Error! C:\jupyter_project\Class\190507~08 파이썬 입문 세미나\04 스크립트\python02\data\file_arr1.npy is not UTF-8 encoded
# 	Saving disabled.
# 	See Console for more details.

# 원인
# - 파일이 ipynb 이 아니라 npy 라서 주피터 노트북으로 편집 안됨.
# - npy 가 UTF-8 로 변환이 안되어서 주피터 노트북으로 편집이 안됨.

# - 편집은 불가하지만 아래 Cell 에서 실행해보면 arr1 불러오는 것은 당연히 가능함.
# - 주피터에서 arr1 을 편집하는 것은 안되지만.

arr1_l = np.load('data/file_arr1.npy')
arr1_l

# file_arr1.npy 불러오는 것은 가능함.

np.savez('data/array_archive.npz', a=arr1, b=arr2)

arch = np.load('data/array_archive.npz')

arch

# npy : array 가 1개일 때??
# npz : array 가 2개일 때??

arch['a']

arch['b']

### 4.2 텍스트 파일 불러오기와 저장하기

arr = np.array( [[2.23342715,-0.37376633,-1.05142871],
                 [-0.57247149,-1.35777871,0.28676036],
                 [-0.01042671,-0.0211314,-0.72049352]])

arr

np.savetxt('data/array_ex.txt', arr, delimiter=',')

arr_l = np.loadtxt('data/array_ex.txt', delimiter=',')

arr_l

---

# 5. 선형대수

x = np.array([[1., 2., 3.], [4., 5., 6.]])

y = np.array([[6., 23.], [-1, 7], [8, 9]])

x

y

x.shape

y.shape

# np.dot(x, y)
x.dot(y)

# numpy.dot 은 numpy array 의 행렬곱과 비슷함

from numpy.linalg import inv

# linalg (linear algebra) 선형대수
# inverse 정반대, 역

X = np.random.randn(5, 5)

X

mat = X.T.dot(X)

mat

inv(mat)  # 행렬을 역행렬으로 변환

# 행렬 * 역행렬 = 단위 행렬
mat.dot(inv(mat))

---

# 6. 난수 생성

samples = np.random.normal(size=(4, 4))

samples

import random

N = 1000000

%time samples = [random.normalvariate(0, 1) for _ in range(N)]

samples[:10]

%time samples_arr = np.random.normal(size=N)

samples_arr[:10]

#### numpy.random 함수

함수 | 설명
:---|:---
seed        | 난수 발생기의 시드를 지정한다.
permutation | 순서를 임의로 바꾸거나 임의의 순열을 반환한다.
shuffle     | 리스트나 배열의 순서를 뒤섞는다.
rand        | 균등분포에서 표본을 추출한다.
randint     | 주어진 최소/최대 범위 안에서 임의의 난수를 추출한다.
randn       | 표준편차가 1이고 평균 값이 0인 정규분표에서 표본을 추출한다.
binomial    | 이항분포에서 표본을 추출한다.
mormal      | 정규분포에서 표본을 추출한다.
beta        | 베타분포에서 표본을 추출한다.
chisquare   | 카이제곱분포에서 표본을 추출한다.
gamma       | 감마분포에서 표본을 추출한다.
uniform     | 균등(0,1)분포에서 표본을 추출한다.


---

# 7. 예제: 계단 오르내리기

import matplotlib.pyplot as plt
%matplotlib inline

# Matplotlib은 파이썬에서 매트랩과 유사한 그래프 표시를 가능케 하는 라이브러리다.

import random

position = 0
walk     = [position]
steps    = 1000

for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)


plt.plot(walk[:100])



nsteps = 1000
draws  = np.random.randint(0, 2, size=nsteps)
steps  = np.where(draws > 0, 1, -1)
walk   = steps.cumsum()


plt.plot(walk[:100])

walk.min()

walk.max()

(np.abs(walk) >= 10).argmax()

np.abs(walk) >= 10

---

# end of file



# Numpy tutorial

## 참고 사이트

[Python Tutorial](https://docs.python.org/3/tutorial/)

[Numpy Tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)

[Numpy Reference](https://docs.scipy.org/doc/numpy/reference/index.html#reference)

[Numpy 참고사이트](http://taewan.kim/post/numpy_cheat_sheet/#2-배열-생성)

---

## 1. The Basics

### 1.1 An example

import numpy as np

a = np.arange(15).reshape(3, 5)
a

a.shape

a.ndim

a.dtype.name

a.itemsize # size(byte) of item

a.size

type(a)

b = np.array([6, 7, 8])
b

type(b)

### 1.2 Array Creation

a = np.array([2,3,4])
a

a.dtype

b = np.array([1.2, 3.5, 5.1])
b.dtype

a = np.array(1,2,3,4)    # WRONG

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-18-4cbd305744dd> in <module>
----> 1 a = np.array(1,2,3,4)    # WRONG
      2 
      3 # []에다가 1,2,3,4 를 넣어줘야 함

ValueError: only 2 non-keyword arguments accepted

[]에다가 1,2,3,4 를 넣어줘야 함

a = np.array([1,2,3,4])  # RIGHT

b = np.array([(1.5,2,3), (4,5,6)])
b

c = np.array( [ [1,2], [3,4] ], dtype=complex )
c

np.zeros( (3,4) )

np.ones( (2,3,4), dtype=np.int16 )

np.empty( (2,3) )

np.arange( 10, 30, 5 )

np.arange( 0, 2, 0.3 )                 # it accepts float arguments

from numpy import pi
np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2

x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points
f = np.sin(x)
f

### 1.3 Printing Arrays

a = np.arange(6)                         # 1d array
print(a)

b = np.arange(12).reshape(4,3)           # 2d array
print(b)

c = np.arange(24).reshape(2,3,4)         # 3d array
print(c)

print(np.arange(10000))

print(np.arange(10000).reshape(100,100))

### 1.4 Basic Operations

a = np.array( [20,30,40,50] )
a

b = np.arange( 4 )
b

c = a-b
c

b**2

10 * np.sin(a)

a < 35

A = np.array( [[1,1],
            [0,1]] )
            
B = np.array( [[2,0],
            [3,4]] )
           

A * B                       # elementwise product

A @ B                       # matrix product

A.dot(B)                    # another matrix product

a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))

a

b

a *= 3
a

b += a
b

a += b                  # b is not automatically converted to integer type

a = np.ones(3, dtype=np.int32)
b = np.linspace(0,pi,3)
b.dtype.name

b

c = a+b
c

c.dtype.name

d = np.exp(c*1j)
d

d.dtype.name

a = np.random.random((2,3))
a

a.sum()

a.min()

a.max()

b = np.arange(12).reshape(3,4)
b

b.sum(axis=0)                            # sum of each column

b.min(axis=1)                            # min of each row

b.cumsum(axis=1)                         # cumulative sum along each row

### 1.5 Universal Functions

B = np.arange(3)
B

np.exp(B)

np.sqrt(B)

C = np.array([2., -1., 4.])

np.add(B, C)

### 1.6 Indexing, Slicing and Iterating

a = np.arange(10)**3
a

a[2]

a[2:5]

a[:6:2] = -1000
a

a[ : :-1]                                 # reversed a

for i in a:
    print(i**(1/3.))

def f(x,y):
    return 10*x+y

b = np.fromfunction(f,(5,4),dtype=int)
b

b[2,3]

b[0:5, 1]                       # each row in the second column of b

b[ : ,1]                        # equivalent to the previous example

b[1:3, : ]                      # each column in the second and third row of b

b[-1]                           # the last row. Equivalent to b[-1,:]

c = np.array( [[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
                [ 10, 12, 13]],
               [[100,101,102],
                [110,112,113]]])
c.shape

c[1,...]                                   # same as c[1,:,:] or c[1]

c[...,2]                                   # same as c[:,:,2]

for row in b:
    print(row)

for element in b.flat:
    print(element)

---

## 2. Shape Manipulation

### 2.1 Changing the shape of an array

a = np.floor(10*np.random.random((3,4)))
a

a.shape

a.ravel()  # returns the array, flattened

a.reshape(6,2)  # returns the array with a modified shape

a.T  # returns the array, transposed

a.T.shape

a.shape

# The reshape function returns its argument with a modified shape,
# whereas the ndarray.resize method modifies the array itself
a

a.resize((2,6))
a

a.reshape(3,-1)

### 2.2 Stacking together different arrays

a = np.floor(10*np.random.random((2,2)))
a

b = np.floor(10*np.random.random((2,2)))
b

np.vstack((a,b))

np.hstack((a,b))

from numpy import newaxis
np.column_stack((a,b))     # with 2D arrays

a = np.array([4.,2.])
a

b = np.array([3.,8.])
b

np.column_stack((a,b))     # returns a 2D array

np.hstack((a,b))           # the result is different

a[:,newaxis]               # this allows to have a 2D columns vector

np.column_stack((a[:,newaxis],b[:,newaxis]))

np.hstack((a[:,newaxis],b[:,newaxis]))   # the result is the same

# In complex cases, r_ and c_ are useful for creating arrays by stacking numbers along one axis.
# They allow the use of range literals (“:”)
np.r_[1:4,0,4]

### 2.3 Splitting one array into several smaller ones

a = np.floor(10*np.random.random((2,12)))
a

np.hsplit(a,3)   # Split a into 3

np.hsplit(a,(3,4))   # Split a after the third and the fourth column

---

## 3. Copies and Views

### 3.1 No Copy at All

a = np.arange(12)
a

b = a            # no new object is created

b is a           # a and b are two names for the same ndarray object

b.shape = 3,4    # changes the shape of a

a.shape

def f(x):
    print(id(x))

id(a)                           # id is a unique identifier of an object

f(a)

### 3.2 View or Shallow Copy

c = a.view()

c is a

c.base is a                        # c is a view of the data owned by a

c.flags.owndata

c.shape = 2,6                      # a's shape doesn't change

a.shape

c[0,4] = 1234                      # a's data changes

a

s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"

s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10

a

### 3.3 Deep Copy

d = a.copy()                          # a new array object with new data is created

d is a

d.base is a                           # d doesn't share anything with a

d[0,0] = 9999

a

---

## 4. Fancy indexing and index tricks

### 4.1 Indexing with Arrays of Indices

a = np.arange(12)**2                       # the first 12 square numbers
a

i = np.array( [ 1,1,3,8,5 ] )              # an array of indices
i

a[i]                                       # the elements of a at the positions i

# index tricks
# a 가 array 인데 a 뒤에 [i]를 붙이니까
# a 가 사용자 함수처럼 됨

j = np.array( [ [ 3, 4], [ 9, 7 ] ] )      # a bidimensional array of indices
j

a[j]

# index tricks
# a 가 array 인데 a 뒤에 [i]를 붙이니까
# a 가 사용자 함수처럼 됨

palette = np.array( [ [0,0,0],                # black
                      [255,0,0],              # red
                      [0,255,0],              # green
                      [0,0,255],              # blue
                      [255,255,255] ] )       # white
                     

image = np.array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
                    [ 0, 3, 4, 0 ]  ] )
                   

palette[image]                            # the (2,4,3) color image

# array([[[  0,   0,   0],      # 0    black
#         [255,   0,   0],      # 1    red
#         [  0, 255,   0],      # 2    green
#         [  0,   0,   0]],     # 0    black

#        [[  0,   0,   0],      # 0    black
#         [  0,   0, 255],      # 3    blue
#         [255, 255, 255],      # 4    white
#         [  0,   0,   0]]])    # 0    black

a = np.arange(12).reshape(3,4)
a

i = np.array( [ [0,1],                     # indices for the first dim of a
                [1,2] ] )
j = np.array( [ [2,1],                     # indices for the second dim
                [3,3] ] )
i

j

a[i,j]                                     # i and j must have equal shape

a[i,2]

# i 가 [[0, 1]
#      [1, 2]] 니까
# 2번째 열인 2, 6, 10 이 
#      [[2, 6]
#      [6, 10]] 이 됨

a[:,j]                                     # i.e., a[ : , j]

l = [i,j]
l

a[l]                                       # equivalent to a[i,j]

s = np.array( [i,j] )
s

a[s]                                       # not what we want

a[tuple(s)]                                # same as a[i,j]

time = np.linspace(20, 145, 5)                 # time scale
time

data = np.sin(np.arange(20)).reshape(5,4)      # 4 time-dependent series
data

ind = data.argmax(axis=0)                  # index of the maxima for each series
ind

time_max = time[ind]                       # times corresponding to the maxima
time_max

data_max = data[ind, range(data.shape[1])] # => data[ind[0],0], data[ind[1],1]...
data_max

np.all(data_max == data.max(axis=0))

a = np.arange(5)
a

a[[1,3,4]] = 0
a

a = np.arange(5)
a

a[[0,0,2]]=[1,2,3]
a

a = np.arange(5)
a

a[[0,0,2]]+=1
a

### 4.2 Indexing with Boolean Arrays

a = np.arange(12).reshape(3,4)
a

b = a > 4
b                                          # b is a boolean with a's shape

a[b]                                       # 1d array with the selected elements

a[b] = 0                                   # All elements of 'a' higher than 4 become 0
a

[망델브로 집합](https://ko.wikipedia.org/wiki/망델브로_집합)

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def mandelbrot( h,w, maxit=20 ):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2         # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime
    

plt.imshow(mandelbrot(400,400))
plt.show()

a = np.arange(12).reshape(3,4)
a

b1 = np.array([False,True,True])             # first dim selection
b2 = np.array([True,False,True,False])       # second dim selection

a[b1,:]                                   # selecting rows

a[b1]                                     # same thing

a[:,b2]                                   # selecting columns

a[b1,b2]                                  # a weird thing to do

### 4.3 The ix_() function

a = np.array([2,3,4,5])
b = np.array([8,5,4])
c = np.array([5,4,6,8,3])

ax,bx,cx = np.ix_(a,b,c)

ax

bx

cx

ax.shape, bx.shape, cx.shape

result = ax+bx*cx
result

result[3,2,4]

a[3]+b[2]*c[4]

def ufunc_reduce(ufct, *vectors):
    vs = np.ix_(*vectors)
    r = ufct.identity
    for v in vs:
        r = ufct(r,v)
    return r

ufunc_reduce(np.add,a,b,c)

---

## 5. Linear Algebra

[<참고>선형대수 함수](https://rfriend.tistory.com/380)

### 5.1 Simple Array Operations

import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)

a.transpose()

np.linalg.inv(a)

u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
u

j = np.array([[0.0, -1.0], [1.0, 0.0]])
j

j @ j        # matrix product

np.trace(u)  # trace

y = np.array([[5.], [7.]])
y

np.linalg.solve(a, y)

np.linalg.eig(j)

---

## 6. Tricks and Tips

### 6.1 Automatic Reshaping

a = np.arange(30)
a

a.shape = 2,-1,3  # -1 means "whatever is needed"
a.shape

# 30개인데 깊이가 2 이니까 5x3 array 가 필요함.
# 5인지 뭐인지 몰라서 -1을 넣었는데 a.shape 구해보니까 5 가 나옴.

a

### 6.2 Vector Stacking

x = np.arange(0,10,2)                     # x=([0,2,4,6,8])
x

y = np.arange(5)                          # y=([0,1,2,3,4])
y

m = np.vstack([x,y])                      # m=([[0,2,4,6,8],
m

xy = np.hstack([x,y])                     # xy =([0,2,4,6,8,0,1,2,3,4])
xy

# horizontal 하게 길게 stack 하라

### 6.3 Histograms

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = np.random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, density=1)       # matplotlib version (plot)
plt.show()

# Compute the histogram with numpy and then plot it
(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
plt.plot(.5*(bins[1:]+bins[:-1]), n)
plt.show()

---

# end of file



