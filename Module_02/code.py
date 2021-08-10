# 1. 파이썬 프로그래밍 소개

### 1.1 파이썬 언어 소개 및 역사
- 파이썬은 널리 쓰이는 범용, 고급 언어이다. 파이썬의 설계 철학은 코드 가독성에 중점을 두고 있으며 파이썬의 문법은 프로그래머가(C와 같은 언어에서 표현 가능한 것보다도) 더 적은 코드로도 자신의 생각을 표현하도록 한다. 파이썬은 프로그램의 크기에 상관없이 명확하게 프로그램 할 수 있는 구성 요소들을 제공한다. - 위키피디아
- 1991년 네덜랜드 국립 연구소의 Guido Van Rossum에 의해 발표 (현 2018년, 27년의 역사)
- “Python“이라는 이름은 코메디 프로그램 “Monty Python’s Flying Circus”에서 유래
- Python의 원래 의미는 그리스 신화에 나오는 거대한 뱀
- 2000년 Python 2 발표
- 2008년 Python 3 발표
    - Python 2의 경우 2020년까지만 Maintenance가 이루어질 예정

### 1.2 파이썬 비교
- 파이썬 vs 펄  
- 파이썬 vs 자바  
- 파이썬 vs R  

### 1.3 프로그램 언어 일반
- 컴퓨터가 돌아가는 구조 와 프로그램의 역할  
- 파이썬의 위치(하나의 프로그램)  
- 입력 --> 처리 --> 출력

### 1.4 파이썬 언어의 특징
- 고급 프로그래밍 언어  
- 인터프리터 방식의 언어( vs 컴파일러 )  
- 인터랙티브 쉘(계산기, cmd, bash, ...)  
- 객체지향 언어  
- 독립적인 실행 환경 제공  

#### <참고> 실행방식에 따른 분류

<center>구분|<center>인터프리터 방식|<center>컴파일러 방식
:----:|:----|:----
장점|프로그램의 이식성이 높다|실행속도가 빠르다
&nbsp;|배우기 쉽다|효율적인 실행 코드가 생성된다
단점|실행속도가 느리다|배우기 어렵다
&nbsp;|실행시 인터프리터가 항상 필요하다|OS에 종속적이다(실행코드의 이식성이 없다)
예|파이썬, 자바스크립트, 쉘스크립트 등|C, C++, Fortran 등

#### <참고> 프로그래밍 언어의 분류

<center>구분|<center>저급 프로그래밍 언어|<center>고급 프로그래밍 언어
:---:|:---|:---
장점|컴퓨터가 직접 이해하므로 실행이 빠르고 강력하다|사람이 이해하기 쉬우므로 프로그램의 작성이 쉽고 작성된 프로그램이 읽기 쉽다
&nbsp;|시스템을 세부적으로 조작 할 수 있다|오류의 수정이 쉽다
단점|사람이 이해하기 어려우며 사용이 어렵다|저급 프로그래밍 언어에 비해 실행 속도가 느리다
&nbsp;|사용범위가 제한적이다|번역이라는 추가 작업이 필요하다
예|기계어, 어셈블리어|C, C++, JAVA, Python, PHP, C# 등

### 1.5 파이썬 패키지(Library) 구조 및 사용
- 패키지(Library), Framework의 차이
- 파이썬의 한계와 C 라이브러리를 통한 극복
- 대부분 C, C++ 로 작성  

### 1.6 패키지 인스톨
- 소스 컴파일  
- 바이너리 인스톨  
- pip(cmd 명령어)  
    - Python Package Index (PyPI) 패키지 관리자

### 1.7 개발환경
- Terminal, Editor
- IPython, jupyter notebook, Anaconda, ...
- jupyter notebook  

#### <참고> 소프트웨어 다운로드 및 설치

[파이썬 다운로드](https://www.python.org/downloads/)

[아나콘다 버전별 다운로드](https://repo.anaconda.com/archive/)

python 3.6 ==> Anaconda3-5.2.0-Linux-x86_64.sh  
<br>
**주의: 설치시 환경변수( PATH ) 설정**

---

# 2. 파이썬 활용

### 2.1 파이썬은 각 분야에서 활용
- 스크립트, 시스템프로그램, 네트위크 프로그램, 웹, 데이터분석, 머신러닝 등
- 처음부터 다 만들기보다는 패키지를 이용(단 학습필요)

### 2.2 데이터분석에서 쓰는 파이썬
- 전통적인 의미의 프로그램은 아님
- 프로그램의 일부만 이용
- 에러처리, 로그 등이 없어도 관계없음
- 직접 눈으로 보면서 확인, 인터렉티브 쉘

---

# 3. Jupyter notebook 활용

### 3.1 Cell
- Code: 파이썬 코드를 실행할 수 있는 블럭
- Markdown: 마크다운 문법이 적용되는 블럭

### 3.2 Mode
- Edit Mode
  - 셀 안의 내용을 편집할 수 있는 상태
  - 셀 위에서 Enter 키를 눌러 셀 내용을 편집할 수 있다.  
<br>
- Command Mode
  - 셀 밖에서 해당 노트북을 명령어로 컨트롤할 수 있는 상태
  - 셀 위에서 ESC 키를 눌러 명령을 내릴 수 있다.
  - `h`를 눌러 도움말을 볼 수 있다.

---

# 기타

### 코딩규약
- 주석
- 들여쓰기
- 상수명, 변수명, 함수명, 클래스명

```python
# 상수
MAX_CNT = 3

# 변수
val_i = 1
_val, __val

# Function
def f_a():
    return

```python
# 상수
MAX_CNT = 3

# 변수
val_i = 1
_val, __val

# Function
def f_a():
    return

# Class
Car, SportCar, Student
```


---

# end of file

# 102. 파이썬 데이터 타입 및 변수

### 1.1 변수 및 대입
- variable
- structure
- class

### 1.2 데이터 타입 종류

<center>타 입|<center>설 명|<center>예
:---|:---|:---
int|정수형 데이터|100
float|소숫점을 포함한 실수|10.25
bool|참/거짓|True
str|문자열|'LG Electronics'
list|리스트, 순서가 있는 배열, 수정/추가/삭제가 가능한 자료 구조|[1, 2, 3, 'a', 'b']
tuple|튜플, 순서가 있는 배열, 수정/추가/삭제가 불가능한 자료 구조|(1, 2, 3, 'a', 'b')
dict|사전, {key: value}로 구성되어있는 자료 구조|{'Math': 99, 'English': 88, 'Korean': 78}
set|집합, {key}로 구성되어있는 자료 구조|{'a', 'b', 'c'}

<참고> None Type

### 1.3 정수(int) 타입
- 10진수
- 2진수
- 8진수
- 16진수

# 10진수
a = 365

# 2진수
b = 0b101101101

# 8진수
c = 0o555

# 16진수
d = 0x16d



### 1.4 실수(float) 타입

fa = 3.14

fb = 3.1415e2



### 1.5 불린(boolean) 타입

ba = True

bb = False

ba and bb



### 1.6 문자열(str) 타입

# 문자열 생성
sa = 'Hello World'
sb = 'one way of writing a string'

sc = "another way"

sd = '''
This is a longer string that
spans multiple lines
'''

se = '''
This is a longer string that
spnas multiple lines
'''

print(se)

print(sd)

# 문자열 카운트

# 문자열, string

sd.count('\n')

se.count('\n')

se.count('\n')

a = 'this is a string!!'
# a[10] = 'f'

b = a.replace('string', 'longer string')
b

b = a.replace('string', 'longer string')

# 문자열 수정
a = 'this is a string'
a[10] = 'f'    # 문자열 수정 불가
b = a.replace('string', 'longer string')
b

# type 변경

# float : 소숫점을 포함한 실수
# float 을 string 으로 type 변경

a = 5.6
s = str(a)
print(s)

# type 변경

a = 5.6
s = str(a)
s

- print(s) 와 s 는 출력결과가 다름

# Escape 문자
# \n \t \\ \' \" etc
s = '12\\34'
print(s)

se = r'this\has\no\special\characters'
print(se)

s = '12//34'
print(s)

se = r'this\has\no\special\characters'
print(se)

s = '12//34'
s

# 문자열 연산
a = 'this is the first half '
b = 'and this is the second half'
a + b

# 문자열 연산
a = '='
a * 50

a = '='
a * 50

import pandas as pd 
 
url="https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv" 
pd.read_csv(url, nrows=10)

# import matplotlib.pyplot as plt 
# plt.plot([1,2,3,4]) 
# plt.ylabel('some numbers') 
# plt.show() 

# Slicing
a = "20010331Rainy"
print(a[:8])
print(a[8:])
# a 변수의 9번째부터 출력하라

template = '{0:.2f} {1:s} are worth US${2:d}'

template.format(4.5560, 'Argentine Pesos', 1)

# 문자열 format
template = '{0:.2f} {1:s} are worth US${2:d}'

template.format(4.5560, 'Argentine Pesos', 1)

# 문자열 합치기
",".join('abcd')
",".join(['a','b','c','d'])

# 문자열 나누기
a = "a:b:c:d"
a.split(':')

# Strip
a = " hi "
a.rstrip()
a.lstrip()
a.strip()

# upper(), lower()
a = " Hi "
print(a.upper())
print(a.lower())

# upper(), lower()
a = " Hi "

print(a.upper())
print(a.lower())

#### 자주 사용하는 문자열 메소드

<center>메소드|<center>설 명
:---|:---
문자열.startswith(prefix)|문자열이 prefix로 시작하는지 검사하여 True/False 반환
문자열.endswith(suffix)|문자열이 suffix로 끝나는지 검사하여 True/False 반환
문자열.format(value)|문자열의 특정 값을 value로 치환한다.
문자열.replace(old, new)|s에 있는 old를 new로 바꾼다.
문자열.split(sep)|sep를 구분자로 하여 문자열을 분할한다.
문자열.strip(chars)|앞이나 뒤에 나오는 공백이나 chars로 지정된 문자들을 제거한다.


### <참고>
- ASCII CODE
- UNICODE

- utf-8
- cp949



### 1.7 리스트(list) 타입

# 리스트 생성
a = list()
a = []
b = [1, 2, 3, 4, 5]
c = ['aaa', 'bbb', 'ccc']
d = [1, 2, 'aaa', 'bbb']
e = [1, 2, ['aaa', 'bbb']]

# 인덱싱 & 슬라이싱
a = [1,2,3,4,5,['a','b','c','d'],7,8,9]

a[0]

a[1]

a[-1]

a[0:4:2]

a[5]

a[5][2]

# pointer
b = a[5]
b[0] = 'k'

a

from copy import deepcopy
c = deepcopy(a[5])
c[1] = 'm'
c

a

1 in a

1 not in a

# function
len(a)
sum(b)
max(b)
min(b)

del(a[0])

# method
a.append(10)
a.sort(10)
a.reverse(10)
a.remove(10)
a.count(10)

# extend()
b = [1,2,3]
b.extend([4,5])
b

**<참고> 함수의 인자로 자주 활용되는 리스트**



### 1.8 튜플(tuple) 타입

# 튜플의 생성
a = tuple()
a = ()
b = (1,)    # 마지막에 콤마 필요
c = (1, 2, 3)
d = 1, 2, 3 # packing & unpacking
e = (1, 2, ('ab', 'cd'), 3, 4)

**튜플은 수정 및 변경 불가**

c[0] = 9

del c[1]

# packing & unpacking
x,y,z = (1,2,3)
print(x,y,z)

def f_a():
    return (4,5,6)

a,b,c = f_a()
print(a,b,c)

x = 1
y = 2

print('Before: ', x, y)

x, y = y, x

print('After : ', x, y)



### 1.9 사전(dict) 타입

# 사전의 생성
d = {1:'Hello World', 'class':'python', 'pi':3.14, 'lst':[1,2,3]}

d[1]
d['class']
d['pi']
d['lst']

# 순차적인 Index 접근 불가
# dict 순서는 hash 순서
d[0]

d.get('x','default value')

# 사전 추가 & 삭제
d['add'] = 'add value'
del d['class']

d

# 수정 가능한 변수는 key 로 사용 불가
dd = {[1,2,3]:'value'}

# keys(), values(), items()
d.keys()
d.values()
d.items()

for k in d.keys():
    print('key: ', k )

'pi' in d

d.clear()

### 1.10 집합(set) 타입

# 집합의 생성
s = set([1,2,3,4])

x = {2,3,2,3,4,5,1}
y = {3,4,5,6,7}

x

x | y    # 합집합
# | 는 or 

x & y    # 교집합

x - y    # 차집합

x ^ y    # 대칭 차집합

x.add(10)
x

x.update([11,12,13])
x

x.remove(13)
x



### 1.11 None 타입

a = None
a is None
b = 5
b is not None

def add_multiply(a, b, c=None):
    result = a + b

    if c is not None:
        result = result * c

    return result

add_multiply(1,2)



### 1.12 타입 변환

a = 1

type(a)

isinstance(a, int)

float(1)

int(3.14)

str(3.14)

float('3.14')

float('Hello World')

tuple([1,2,3])

list((1,2,3))



### 1.13 변수 할당

a = [1,2,3]
b = [1,2,3]
c = a
d = a[:]    # Numpy, Pandas 에서는 View를 제공(메모리의 효율적 사용)

a is c

a is b

a is d

a == b

id(a)
id(b)
id(c)
id(d)

list = [1,2,3]
id(list[0])

### 1.14 변수 활용

students = [
               {'num':'1','name':'김철수','kor':90,'eng':80,'math':85,'total':0,'avg':0.0,'order':0 },
               {'num':'2','name':'박제동','kor':90,'eng':85,'math':90,'total':0,'avg':0.0,'order':0 },
               {'num':'3','name':'홍길동','kor':80,'eng':80,'math':80,'total':0,'avg':0.0,'order':0 }
           ]

students[0]

students[1]['kor']



---

# end of file

# 103. 연산자

### 1.1 수치 연산자

2 + 3

3 - 2

2 * 3

5 / 2

5 // 2

# // 는 몫

5 % 2

# % 는 나머지

3.14 ** 2

2 ** 0.5
# ** 은 제곱이니까 결국 루트 씌운 것과 같은 말



### 1.2 대입 연산자

a = 20

a += 5
a

a -= 10
a

a *= 2
a

a /= 3
a

a = 3
b = 2
a *= 2 + b    # a = a * ( 2 + b )
a



### 1.3 비교 연산자

a = 95

a == 90

a != 90

a > 90

a < 90

a >= 90

a <= 90

a >= 90 and a <= 100

90 <= a <= 100

100 <= a <= 110

70 <= a <= 80

lst1 = [1,2,'Hello World']
lst2 = [1,2,'Hello World']

id(lst1) == id(lst2)

lst1 == lst2

lst1 != lst2



### 1.4 논리 연산자

True and False

True or False

not True



### 1.5 식별 연산자

lst1 = [1,2,'Hello World']
lst2 = [1,2,'Hello World']

lst1 is lst2

lst1 is not lst2

lst1 == lst2

id(lst1) == id(lst2)



### 1.6 구성원 연산자

lst = [1,2,'Hello World']

1 in lst

3 not in lst

dic = {'a':1, 'b':2, 'c':3 }

'a' in dic

'k' in dic

1 in dic



### 1.7 연산자 활용

students = [
               {'num':'1','name':'김철수','kor':90,'eng':80,'math':85,'total':0,'avg':0.0,'order':0 },
               {'num':'2','name':'박제동','kor':90,'eng':85,'math':90,'total':0,'avg':0.0,'order':0 },
               {'num':'3','name':'홍길동','kor':80,'eng':80,'math':80,'total':0,'avg':0.0,'order':0 }
           ]

# 학생별 합계 구하기
students[0]['total'] = students[0]['kor'] + students[0]['eng'] + students[0]['math']
students[1]['total'] = students[1]['kor'] + students[1]['eng'] + students[1]['math']
students[2]['total'] = students[2]['kor'] + students[2]['eng'] + students[2]['math']

print(students[0]['total'])
print(students[1]['total'])
print(students[2]['total'])

# 학생별 평균 구하기
students[0]['avg'] = students[0]['total'] / 3
students[1]['avg'] = students[1]['total'] / 3
students[2]['avg'] = students[2]['total'] / 3

print(students[0]['avg'])
print(students[1]['avg'])
print(students[2]['avg'])

# 학급 평균 구하기
class_avg = (students[0]['total'] + students[1]['total'] + students[2]['total']) / 3

print(class_avg)

# 과목별 평균 구하기
kor_avg  = (students[0]['kor']  + students[1]['kor']  + students[2]['kor'])  / 3
eng_avg  = (students[0]['eng']  + students[1]['eng']  + students[2]['eng'])  / 3
math_avg = (students[0]['math'] + students[1]['math'] + students[2]['math']) / 3

print(kor_avg)
print(eng_avg)
print(math_avg)

#### <참고> 연산자 우선순위

1 + 2 * 3 / 4



---

# 2. 흐름 제어

### 2.1 흐름과 흐름 제어

print('Hello World')
print(1 + 2)
print('안녕하세요 파이썬')
print('흐름과 흐름 제어 입니다')

### 2.2 선택 흐름과 if 문

#### if 문만 있는 경우

grade = float(input('총 평점을 입력해 주세요: '))

if grade >= 4.3:
    print('당신은 장학금 수여 대상자 입니다.')
    print('축하합니다.')

print('공부 열심히 하세요.')

grade = float(input('총 평점?'))

# 실수 float
# 숫자를 문자형으로 했다가 다시 정수 나 실수로 변경할때는 int , float를 사용하여 변경할 수있습니다. 이런 것을 형변환이라고 합니다.

if grade >= 2.0:
    print('당신은 대빵')
    print('우앙')

#### if else 문

data = int(input('숫자를 입력하시오: '))

if data % 2 == 0:
    print('입력된 값은 짝수입니다.')
else:
    print('입력된 값은 홀수입니다.')

data = int(input('숫자를 입력하시오: '))

if data % 2 == 0:
    print('입력된 값은 짝수입니다.')
else:
    print('입력된 값은 홀수입니다.')


score = int(input('점수를 입력하시오: '))

if score >= 70:
    print('당신은 시험을 통과했습니다.')
else:
    print('당신은 시험을 통과하지 못했습니다.')
    
print('공부 열심히 하세요!')


score = int(input('점수를 입력하시오: '))

if score >= 70:
    print('당신은 시험을 통과했습니다.')
else:
    print('당신은 시험을 통과하지 못했습니다.')
    print('공부 열심히 하세요!')

#### 중첩된 if ~ else ~ 구문

age = int(input('나이를 입력하시오: '))
height = int(input('키를 입력하시오: '))

if age >= 40:
    if height >= 170:
        print('키가 보통 이상 입니다.')
    else:
        print('키가 보통입니다.')
else:
    if height >= 175:
        print('키가 보통 이상 입니다.')
    else:
        print('키가 보통입니다.')


age = int(input('나이를 입력하시오: '))
height = int(input('키를 입력하시오: '))

if age >= 40:
    if height >= 170:
        print('키가 보통 이상 입니다.')
    else:
        print('키가 보통입니다.')

#### if ~ elif ~ 구문

score = int(input('총점을 입력해 주세요: '))

if score >= 90:
    print('수')
else:
    if 80 <= score < 90:
        print('우')
    else:
        if 70 <= score < 80:
            print('미')
        else:
            if 60 <= score < 70:
                print('양')
            else:
                print('가')
        

score = int(input('총점을 입력해 주세요: '))

if score >= 90:
    print('수')
else:
    if 80 <= score < 90:
        print('우')
    else:
        if 70 <= score <80:
            print('미')
        else: 
            if 60 <= score < 70:
                print('양')
            else:
                print('가')

score = int(input('총점을 입력해 주세요: '))

if score >= 90:
    print('수')
elif 80 <= score < 90:
    print('우')
elif 70 <= score < 80:
    print('미')
elif 60 <= score < 70:
    print('양')
else:
    print('가')
        

score = int(input('총점을 입력해 주세요: '))

if score >= 90:
    print('수')
elif 80 <= score < 90:
    print('우')
elif 70 <= score < 80:
    print('미')

#### 조건부 표현식 ( 3항 연산자 )

score = int(input('총점을 입력해 주세요: '))

if score >= 60:
    message = "success"
else:
    message = "failure"
    
print(message)

# 위 항의 수우미양가는 ''써야하고, message 는 변수니까 '' 안 씀.

score = int(input('총점을 입력해 주세요: '))

message = "success" if score >= 60 else "failure"

print(message)

score = int(input('총점을 입력해 주세요: '))

message = "A+" if score >= 90 else 'A0' if score >= 80 else "B+"

print(message)

### 2.3 반복 흐름과  for, while 문

#### for 문

message = 'Hello!'
messages = ['Hello World', '파이썬 문자열입니다']
numbers = (1,2,3)
polygon = {'triangle': 2, 'rectangle': 3, 'line': 1}
color = {'red', 'green', 'blue'}

for item in messages:
    print(item)

# for 루프(loop) 구현

# 파이썬에서도 for반복문을 사용해 반복문을 만들 수 있습니다. 먼저 가장 기본적인 방법은 아래와 같습니다.

sites = ['web', 'is', 'free']
for item in sites:
  print(item)

# for문의 인덱스 순서 값 가져오기

# enumerate : 열거하다

# for문에서 몇 번째 루프인지 index값을 가져오려면 어떻게 할까요? 이 경우 enumerate()를 사용합니다.
# 이 값을 적용하면 첫 번째 인자로 index 순서를 0부터 반환하게 됩니다. 이제 아래와 같이 다시 코드를 작성해보도록 하겠습니다.

sites = ['web', 'is', 'free']
for index, item in enumerate(sites):
  print(str(index) + ' - ' + item)

total = 0

for item in [1,2,3,4,5,6,7,8,9,10]:
    total = total + item
    
print('1부터 10까지 합은 ', total, ' 입니다.')


#### range() 함수

total = 0

for item in range(1,11):
    total = total + item
    
print('1부터 10까지 합은 ', total, ' 입니다.')


total = 0

for item in range(0,101,2):
    total = total + item
    
print('1부터 100까지 짝수 합은 ', total, ' 입니다.')


total = 0

for item in range(1,101,2):
    total = total + item
    
print('1부터 100까지 홀수 합은 ', total, ' 입니다.')

range(1,101,2)

total = 0

for item in range(1,101):
    if (item % 3) == 0 or (item % 7) == 0:
        total = total + item
    
print('1부터 100까지에서 3 혹은 7의 배수의 합은 ', total, ' 입니다.')


total = 0

for item in range(1,101):
    total = total + item
else:
    print('덧셈 작업이 끝났습니다.')
    
print('1부터 100까지 합은 ', total, ' 입니다.')


for i in range(3, 5):

    print('--{}단 시작--'.format(i))
    
    for j in range(1, 10):
    
        print(i, "*", j, "=", i * j)


# 구구단
print("★ 구구단을 출력합니다.\n")
for x in range(2, 10):
    print("------- [" + str(x) + "단] -------")
    for y in range(1, 10):
        print(x, "X", y, "=", x*y)
print("---------------------")

# 구구단 만들기
# 1. for 문

for i in range(1,10):
    for j in range(1,10):
 
        print("{} * {} = {}".format(i, j, i*j))

# 구구단 만들기
# 2. while 문

# while문도 반복문이기 때문에 동일하게 할 수 있다. 9까지 반복시켜야 되기 때문에 10보다 작은 조건에서 반복되게 하면 된다.
 
i = 2
 
while(i<10):
    j = 1 
    while(j<10):
        print("{} * {} = {}".format(i, j, i*j))
        j+=1
    i+=1

# 구구단 만들기
# 3. list를 이용한 for문

# 리스트 자료형도 for문을 돌릴 수 있기 때문에, 리스트를 만들어서 for문을 돌려봤다.
 
num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
 
for i in num_list:
    for j in num_list:
 
        print("{} * {} = {}".format(i, j, i*j))

# 구구단 만들기
# 4. numpy의 ndarray 자료형으로 만든 for문
# 위의 예와 비슷하게 numpy자료형을 만들어서도 할 수 있다.

import numpy as np
 
# num_list = np.arrange(1, 10, 1)
 
for i in num_list:
    for j in num_list:
 
        print("{} * {} = {}".format(i, j, i*j))

# 구구단 만들기
# 5. 리스트와 for문 한 줄에 쓰기

# 파이썬에서는 리스트와 for문을 한 줄에 작성할 수 있다.
# 처음에는 어색하지만 자주 사용하다보면 굉장히 편리한 것을 알 수 있다.
# if문과 for문을 한 줄에 쓰는 문법에 대해서는 아래 포스팅을 참조해보자.

temp = [[j , i , j*i ] for j in range(1,10) for i in range(1,10)]
 
for i in temp:
    print("{} * {} = {}").format(i[0], i[1], i[2])

#### 리스트 내포(List comprehension)

lst = [1,2,3,4]

result = []
for num in lst:
    result.append(num*3)

print(result)

# append() 함수는 리스트 형태의 Data 에 마지막에 하나를 추가하는 함수임

# list comprehension

lst = [1,2,3,4]
result = [num * 3 for num in lst]

print(result)

# 파이썬 문법 규칙, 위 행과 동일한 의미

# list comprehension with if

lst = [1,2,3,4]
result = [num * 3 for num in lst if num % 2 == 0]

print(result)

# 짝수일 경우

#### while 문

count = 1

while count <= 10:
    print(count)
    count = count + 1


count = 1
total = 0

while count <= 100:
    total = total + count
    count = count + 1

print('1부터 100까지의 합은:', total )


count  = 1
result = 0

while count <= 100:
    result = result + count
    count  = count + 1
else:
    print('덧셈이 작업 완료 되었습니다.')

print('1부터 100까지의 합은:', result )


### 2.4 break 문과 continue 문

break_letter = input('중단할 문자를 입력하시오: ')

for letter in 'python':
    if letter == break_letter:
        break
    print(letter)
    
else:
    print('모든 문자 출력 완료!')

break_letter = input('중단할 문자를 입력하시오: ')

for letter in 'python':
    if letter == break_letter:
        break
    print(letter)
    
else:
    print('모든 문자 출력 완료!')

continue_letter = input('건너뛸 문자를 입력하시오: ')

for letter in 'python':
    if letter == continue_letter:
        continue
    print(letter)
    
else:
    print('모든 문자 출력 완료')
# continue 는 넘어가겠다는 의미

### 2.5 pass 문

input_letter = input('중단할 문자를 입력하시오: ')

for letter in 'python':
    if letter == input_letter:
        pass
    print(letter)

# 파이썬 pass
# 아무 기능이 없는 파이썬 제어문
# 그냥 지나가는 문장임
# break(블럭 탈출)의 반대 의미라고 볼 수 있겠다.
# 주로 미구현 블럭에서 들여쓰기로 인한 문제를 방지하기 위해 사용한다.

# 파이썬 pass와 continue 차이점
# pass는 실행 할 것이 아무 것도 없다는 것을 의미. 따라서 아무런 동작을 하지 않고 다음 코드를 실행
# continue는 다음 순번의 loop 실행

for i in range(1, 5):
    if i == 3:
        pass
    print(i)
    
# → i의 값이 3일 때 if 조건에서 아무 것도 실행하지 않고 다음 단계로 넘어가므로 print() 함수를 실행함

for i in range(1, 5):
    if i == 3:
        continue
    print(i)

# → i의 값이 3일 때 곧장 loop의 다음 순번으로 이동하기 때문에 print() 함수가 실행되지 않아 숫자 3이 출력되지 않음    

a = 0

for a in range(1,11):
    pass
    print(a)
# range 안에 0 이 없지만 그냥 pass 하고 1~10까지 출력하라는 의미

a = 0

for a in range(1,11):
    continue
    print(a)
    
# range 안에 0 이 없으므로 print 할 것이 없음

### 2.6 무한 반복

choice = None

while True:
    print('1. 원 그리기')
    print('2. 사각형 그리기')
    print('3. 선 그리기')
    print('4. 종료')
    
    choice = input('메뉴를 선택하시오: ')
    
    if choice == '1':
        print('원 그리기를 선택했습니다.')
    elif choice == '2':
        print('사각형 그리기를 선택했습니다.')
    elif choice == '3':
        print('선 그리기를 선택했습니다.')
    elif choice == '4':
        print('종료합니다.')
        break
    else:
        print('잘못된 선택을 했습니다.')
        
# while True 는 break 안 걸어주면 무한 반복함
# While True 로 W 대문자면 오류남

# 문제) 2부터 차례대로 소수를 모두 더했을 때, 그 소수의 합계가 125000이 넘었을 때의 소수는?

pn_sum = 0
i = 2
while True:
    number = i
    count = 0
    for i in range(1, number+ 1):
        if number%i == 0:
            count = count +1
            pass
        pass
    if count == 2:
        pn_sum = pn_sum + number
        if pn_sum >= 125000:            # 멈추기 위한 조건
            print("answer is {}".format(number))
            break
        pass
    i = i + 1
    pass

# 문제) 2부터 차례대로 소수를 모두 더했을 때, 그 소수의 합계가 125000이 넘었을 때의 소수는?

# Error문

pn_sum = 0
while True:
    number = i
    count = 0
    for i in range(1, number+ 1):
        if number%i == 0:
            count = count +1
            pass
        pass
    if count == 2:
        pn_sum = pn_sum + number
        if pn_sum > 125000:
            break
print(pn_sum)

# Error : i 에 대한 로직이 없다. i 를 설정해줘야 함. 

# 문제) 숫자 123456을 한 자리씩 분리하여 가로로 출력하고, 합계를 분리된 숫자 아래에 출력하시오.

number = 123456
sum = 0
while True:
        split_number = number%10
        number = number//10
        sum = sum + split_number
        print(split_number, end = " ")
        if number == 0 :
            break
        pass
print()
print(sum)
pass

# 문제) 숫자 123456을 한 자리씩 분리하여 가로로 출력하고, 합계를 분리된 숫자 아래에 출력하시오.

# Error문

number = 123456
sum = 0
while True:
        split_number = number%10
        number = number//10
        sum = sum + split_number
        print(split_number, end = " ")
        if split_number == 1 :     # 내가 처리하고 있는게 마지막이냐
            break
        pass
print()
print(sum)
pass

### 2.7 예제: 요일 구하기

1. 서기 1년 1월 1일은 월요일이다.  
<br>
1. 윤년을 구하는 공식은 다음과 같다.
  - 4로 나누어지는 해는 윤년이다.
  - 100으로 나누어지는 해는 윤년이 아니다.
  - 400으로 나누어지는 해는 윤년이다.

# 년월일을 입력 받는다.
year  = int(input('년도를 입력하시오: '))
month = int(input('월을 입력하시오: '))
day   = int(input('일을 입력하시오: '))

total_days = 0

# 달별 날수를 리스트로 저장
year_month_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]


# 서기 1년부터 year-1 까지의 각 년도 별 날수를 합한다.
for item in range(1, year):
    if item % 400 == 0:                 # 400으로 나뉘는 해: 윤년
        total_days = total_days + 366
    elif item % 100 == 0:               # 100으로 나뉘는 해: 평년
        total_days = total_days + 365
    elif item % 4 == 0:                 # 4으로 나뉘는 해  : 윤년
        total_days = total_days + 366
    else:
        total_days = total_days + 365


# 1월 달부터 month-1 까지의 각 달의 날수를 합한다.
for item in range(1, month):
    total_days = total_days + year_month_days[item]
    

# 입력된 달이 3이상이고 해당년도가 윤년일 경우 1을 추가
if month >= 3:
    if year % 400 == 0:                 # 400으로 나뉘는 해: 윤년
        total_days = total_days + 1
    elif year % 100 == 0:               # 100으로 나뉘는 해: 평년
        total_days = total_days + 0
    elif year % 4 == 0:                 # 4으로 나뉘는 해  : 윤년
        total_days = total_days + 1
    else:
        total_days = total_days + 0


total_days += day

# 총 날수를 7로 나눈 나머지를 구한다.
remainder = total_days % 7

if remainder == 0:
    print('일요일 입니다.')
elif remainder == 1:
    print('월요일 입니다.')
elif remainder == 2:
    print('화요일 입니다.')
elif remainder == 3:
    print('수요일 입니다.')
elif remainder == 4:
    print('목요일 입니다.')
elif remainder == 5:
    print('금요일 입니다.')
elif remainder == 6:
    print('토요일 입니다.')


### 2.8 예제: 성적 처리 시스템

# 학생 데이터 초기화
students = [
               {'num':'1','name':'김철수','kor':90,'eng':80,'math':85,'total':0,'avg':0.0,'order':0 },
               {'num':'2','name':'박제동','kor':90,'eng':85,'math':90,'total':0,'avg':0.0,'order':0 },
               {'num':'3','name':'홍길동','kor':80,'eng':80,'math':80,'total':0,'avg':0.0,'order':0 }
           ]


# 반 총점, 평균 및 각 과목별 총점과 평균 초기화
class_total = 0
class_avg   = 0.0
kor_total   = 0
kor_avg     = 0.0
eng_total   = 0
eng_avg     = 0.0
math_total  = 0
math_avg    = 0.0


# 학생들의 성적 총점과 평균 및 반 총점과 과목별 총점을 구한다.
for student in students:
    student['total'] = student['kor'] + student['eng'] + student['math']
    student['avg']   = student['total'] / 3
    
    class_total = class_total + student['total']
    kor_total   = kor_total   + student['kor']
    eng_total   = eng_total   + student['eng']
    math_total  = math_total  + student['math']

class_avg = class_total / len(students)
kor_avg   = kor_total   / len(students)
eng_avg   = eng_total   / len(students)
math_avg  = math_total  / len(students)


# 학생별 성적 처리 결과를 출력한다.
for student in students:
    print('번호: {:s}, 이름: {:s}, 국어: {:d}, 영어: {:d}, 수학: {:d}, 총점: {:d}, 평균: {:.2f}'.format(
           student['num'], student['name'],
           student['kor'], student['eng'], student['math'],
           student['total'], student['avg'])
         )
print()


# 반 평균 및 각 과목별 평균을 출력한다.
print('반 평균  : {:.2f}'.format(class_avg))
print('국어 평균: {:.2f}'.format(kor_avg)  ) 
print('영어 평균: {:.2f}'.format(eng_avg)  )
print('수학 평균: {:.2f}'.format(math_avg) )


### 2.9 예제: 모스부호

# 모스부호
dic = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.':'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z'
}
 
# 풀어야할 암호: HE SLEEPS EARLY
code = '.... .  ... .-.. . . .--. ...  . .- .-. .-.. -.--'


for word in code.split("  "):
    for char in word.split():
        print(dic.get(char), end='')
    print(" ", end='')

### 2.10 예제: 게시글 제목 추출

# 게시글 제목
title = """On top of the world! Life is so fantastic if you just let it.
I have never been happier. #nyc #newyork #vacation #traveling"""

tag_list = []
for tag in title.split(" "):
    if tag.startswith("#"):
        tag_list.append(tag[1:])

# 문자열.split(sep) : sep를 구분자로 하여 문자열을 분할한다.
# title 이라는 문장에서 " "를 구분자로 하여 각각의 문자열을 분할함. 
# "#" 으로 시작하는 문자열이 있다면
# append 하라는 뜻

        # >>> a = [1, 2, 3, 4, 5]
        # >>> a.append(6)
        # >>> a
        # [1, 2, 3, 4, 5, 6]
        
print(tag_list)

### 2.11 예제: 숫자 야구, 숫자 Up & Down, 로또

# 예제) 숫자 야구게임

import random    
    # 게임을 위한 랜덤 숫자 생성
rn = ["0", "0", "0"]
rn[0] = str(random.randrange(1, 9, 1))
rn[1] = rn[0]
rn[2] = rn[0]
while (rn[0] == rn[1]):
    rn[1] = str(random.randrange(1, 9, 1))
while (rn[0] == rn[2] or rn[1] == rn[2]):
    rn[2] = str(random.randrange(1, 9, 1))

    #print(rn)

    t_cnt = 0 # 시도횟수
    s_cnt = 0 # 스트라이크 갯수
    b_cnt = 0 # 볼 갯수

    print("숫자야구게임을 시작합니다 !!!")
    print("---------------------------")
    while ( s_cnt < 3 ):

        num = str(input("숫자 3자리를 입력하세요 : "))

        s_cnt = 0
        b_cnt = 0

        for i in range(0, 3):
            for j in range(0, 3):
                if(num[i] == str(rn[j]) and i == j):
                    s_cnt += 1
                elif(num[i] == str(rn[j]) and i != j):
                    b_cnt += 1
        print("결과 : [", s_cnt, "] Strike [", b_cnt, "] Ball")
        t_cnt += 1
    print("---------------------------")
    print(t_cnt, "번 만에 정답을 맞추셨습니다.")

# 예제) 숫자 Up & Down 게임

import random

# 게임을 위한 랜덤 숫자 생성

rn = random.randrange(1, 101, 1)
num = -1

t_cnt = 0 # 시도횟수

print("1~100 숫자 Up & Down 게임을 시작합니다 !!!")
print("---------------------------")
while ( rn != num ):

    num = int(input("1 ~ 100 사이의 숫자를 입력하세요 : "))

    if (num > rn):
        print("Down")
    elif (num < rn):
        print("Up")

    t_cnt += 1
print("---------------------------")
print(t_cnt, "번 만에 정답을 맞추셨습니다.")

# 예제) Lotto 자동번호 추출

import random

num = int(input("lotto 게임 수를 입력하세요 : "))

print("lotto 자동 번호 입니다.")
print("----------------------")
# 입력한 게임 수 만큼 반복
for x in range(1, num+1):
    lotto = [0, 0, 0, 0, 0, 0]

    lotto[0] = random.randrange(1, 46, 1)

    lotto[1] = lotto[0]
    lotto[2] = lotto[0]
    lotto[3] = lotto[0]
    lotto[4] = lotto[0]
    lotto[5] = lotto[0]

    # 중복된 수가 발생되지 않도록 채번
    while (lotto[0] == lotto[1]):
        lotto[1] = random.randrange(1, 46, 1)
    while (lotto[0] == lotto[2] or lotto[1] == lotto[2]):
        lotto[2] = random.randrange(1, 46, 1)
    while (lotto[0] == lotto[3] or lotto[1] == lotto[3] or lotto[2] == lotto[3]):
        lotto[3] = random.randrange(1, 46, 1)
    while (lotto[0] == lotto[4] or lotto[1] == lotto[4] or lotto[2] == lotto[4] or lotto[3] == lotto[4]):
        lotto[4] = random.randrange(1, 46, 1)
    while (lotto[0] == lotto[5] or lotto[1] == lotto[5] or lotto[2] == lotto[5] or lotto[3] == lotto[5] or lotto[4] == lotto[5]):
        lotto[5] = random.randrange(1, 46, 1)

    # 결과를 정렬
    lotto.sort()

    # 결과 출력
    print(lotto)

# 예제) (*)로 Tree 만들기

 line = int(input("Tree 의 높이를 입력하세요(5~30) : "))

    for x in range(1, line * 2, 2):
        print((" " * ( (line * 2 - 1 - x) // 2 )) + ("*" * x))

    for y in range(1, 4):
        print(" " * (line-2) + "***")

# 예제) (*)로 Diamond 만들기

line = int(input("Diamond 의 길이를 입력하세요(2~30) : "))

    for x in range(1, line * 2, 2):
        print((" " * ( (line * 2 - 1 - x) // 2 )) + ("*" * x))

    for y in range(line * 2-3, 0, -2):
        print((" " * ( (line * 2 - 1 - y) // 2 )) + "*" * y)

---

# 기타

### enumerate()

some_list = ['foo', 'bar', 'baz']
for i, v in enumerate(some_list):
    print('i: {}, v: {}'.format(i,v))


t = [1, 5, 7, 33, 39, 52]

for p in enumerate(t):
    print(p)

for i, v in enumerate(t):
    print("index : {}, value: {}".format(i,v))

### zip()

# zip(*iterable)은 동일한 개수로 이루어진 자료형을 묶어 주는 역할을 하는 함수이다.

# ※ 여기서 사용한 *iterable은 반복 가능(iterable)한 자료형 여러 개를 입력할 수 있다는 의미이다.

    # >>> list(zip([1, 2, 3], [4, 5, 6]))
    # [(1, 4), (2, 5), (3, 6)]
    # >>> list(zip([1, 2, 3], [4, 5, 6], [7, 8, 9]))
    # [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
    # >>> list(zip("abc", "def"))
    # [(’a’, ’d’), (’b’, ’e’), (’c’, ’f’)]

# zip : zip(*iterable)은 iterable한 객체를 인수로 받으며 동일한 개수로 이루어진 자료형을 묶어서 iterator로 반환.

    # >>> a = zip([1,2,3], (4,5,6))
    # >>> next(a)
    # (1, 4)
    # >>> next(a)
    # (2, 5)
    # >>> next(a)
    # (3, 6)

# zip은 두개의 iterable객체를 묶어 for문을 한꺼번에 반복시킬때 유용합니다.

    # >>> country = ['대한민국','스웨덴', '미국']
    # >>> capital = ['서울','스톡홀름','워싱턴']
    # >>> for coun, cap in zip(country, capital):
    # ...     print('국가명 : {}, 수도:{}'.format(coun,cap))
    # ... 
    # 국가명 : 대한민국, 수도:서울
    # 국가명 : 스웨덴, 수도:스톡홀름
    # 국가명 : 미국, 수도:워싱턴

seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']

for a, b in zip(seq1, seq2):
    print('a: {}, b: {}'.format(a, b))


### 리스트, 사전 내포(comprehension)

# len : Str함수는 아니며, 일반 내장함수입니다. 문자열 길이 또는 컬렉션형의 길이를 나타냅니다.

    # >>> len("abcd12345abcdefg")
    # 16

# 리스트 길이구하기
# 리스트 길이를 구하기 위해서는 다음처럼 len 함수를 사용해야 한다.

    # >>> a = [1, 2, 3]
    # >>> len(a)
    # 3

    # >>> a = "Life is too short"
    # >>> len(a)
    # 17

# 소문자를 대문자로 바꾸기(upper)
# upper 함수는 소문자를 대문자로 바꾸어 준다. 만약 문자열이 이미 대문자라면 아무 변화도 일어나지 않을 것이다.
    # >>> a = "hi"
    # >>> a.upper()
    # ’HI’

# 대문자를 소문자로 바꾸기(lower)
# lower 함수는 대문자를 소문자로 바꾸어 준다.
    # >>> a = "HI"
    # >>> a.lower()
    # ’hi’

strings = ['a', 'as', 'bat', 'car', 'dove', 'python']

# list
lst = [x.upper() for x in strings if len(x) > 2]
lst

# dict
dic = {key: index for index, key in enumerate(strings)}
dic

# enumerate는 “열거하다”라는 뜻이다.
# 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 돌려준다.

# ※ 보통 enumerate 함수는 다음 예제처럼 for문과 함께 자주 사용한다.

    # >>> for i, name in enumerate([’body’, ’foo’, ’bar’]):
    # ... print(i, name)
    
    # 0 body
    # 1 foo
    # 2 bar
    
# 순서 값과 함께 body, foo, bar가 순서대로 출력되었다.
# 즉 위 예제와 같이 enumerate를 for문과 함께 사용하면 자료형의 현재 순서(index)와 그 값을 쉽게 알 수 있다.
# for문처럼 반복되는 구간에서 객체가 현재 어느 위치에 있는지 알려 주는 인덱스 값이 필요할때
# enumerate 함수를 사용하면 매우 유용하다.


---

# end of file



# 104. 함수(Function)

### 1.1 사용자 정의 함수

def say():
    print('Hello World')
    print('Hello World')
    print('Hello World')

say()

#### 함수의 입력값(인자)

# 입력값
def say_n( cnt ):
    for i in range(cnt):
        print(i, 'Hello World')

say_n(3)

#### 함수의 출력값(리턴값)

def f1(x):
    a = 3
    b = 5
    y = a*x + b
    return y

c = f1(10)
# x 에 10을 넣은 값 35를 c 라는 변수에 반환함. 
# c 값을 출력하면 반환(return)된 값이 없음.

print(c)

def f2(x):
    a = 3
    b = 5
    y = a*x + b
    print(y)

d = f2(10)
print(d)

# x 에 10을 넣은 값 35를 d 라는 변수에 반환(return)하지 않음. 
# d 값을 출력하면 반환(return)된 값이 없음.

# 출력값(return)
def add( a, b ):
    result = a + b
    return result


res = add( 2, 3 )
print(res)

#### 여러개의 변수를 리턴할 수 있다

# 여러개의 출력값(return)
def add_mul( a, b ):
    result_add = a + b
    result_mul = a * b
    return result_add, result_mul


res = add_mul(3,4)
res1, res2 = add_mul(3,4)

print(res)
print(res1, res2 )

#### 입력값: 리스트와 숫자의 차이

# 입력값: list, number
def add_lst(num1, lst1):
    num1 = num1 + 1
    lst1.append(1)
    
# append 는 리스트에 원소 추가하는 것. 
# lst1 이라는 리스트에 원소 1을 추가하라는 뜻.

num = 1
lst = [1,2,'Hello World']
print('Before: ', num, lst)

add_lst( num, lst)
print('After : ', num, lst)

#### Global 변수

# global 변수
def add_one(x):
    global a
    a = a + 1
    
# 기본서 "왕초보", "지역변수", "전역변수" 검색

a = 1
add_one(a)
print(a)

### 1.2 일반인자(순서), 키워드인자(변수명)

# 일반인자, 키워드인자
def sub_arg( a, b ):
    result = a - b
    return result

res1 = sub_arg( 4, 2 )
res2 = sub_arg( b = 4, a = 2 )
print(res1)
print(res2)

### 1.3 기본인자(Default)

# 기본인자
def circle_area(r, pi=3.14):
    area = pi * (r ** 2)
    return area

res1 = circle_area(3)
res2 = circle_area(3, 3.1415)

print('면적: {:.4f}'.format(res1))
print('면적: {:.4f}'.format(res2))

### 1.4 가변인자

# 가변인자
def add_many(*args): 
    result = 0 
    for i in args: 
        result = result + i 
        
    return result

# args는 매개변수를 뜻하는 영어 단어 arguments의 약자이며 관례적으로 자주 사용한다.

add_many(1,2,3,4,5,6,7)

def add_mul(choice, *args): 
    if choice == "add": 
        result = 0 
        for i in args: 
            result = result + i 
    elif choice == "mul": 
        result = 1 
        for i in args: 
            result = result * i 
            
    return result 

res1 = add_mul('add', 1,2,3,4)
res2 = add_mul('mul', 1,2,3,4)

print(res1)
print(res2)

### 1.5 정의되지 않은 인자

# 정의되지 않은 인자
def print_kwargs(**kwargs):
    print(kwargs)
    
# 여기에서 kwargs는 keyword arguments의 약자이며 args와 마찬가지로 관례적으로 사용한다.

print_kwargs(a=1)
print_kwargs(name='foo', age=3)

def area_arg(r, *pi, **info):
    for item in pi:
        area = item * (r ** 2)
        print('반지름: ', r, 'PI: ', item, '면적: ', round(area,2) )
        
    for key in info:
        print(key, ':', info[key])

area_arg(3, 3.14, 3.1415, line_color='파랑', area_color='노랑')

print()

area_arg(5, 3.14, 3.1415, polygon_name='원', value='면적')


#### <참고> tuple, dict의 활용

def f_a(a,b,c):
    print(a,b,c)

def f_b(a,b,c):
    print(a,b,c)
   

lst = [1,2,3]
f_a(*lst)

dic = {'a':4, 'b':5, 'c':6}
f_b(**dic)

### 1.6 익명함수( lambda )

add = lambda a, b: a + b
result = add(3, 4)
print(result)

def f_c( a, b, func ):
    result = func(a,b)
    return result

res = f_c(2,3, lambda x, y: x + y )
print(res)

### 1.7 예제: 요일 구하기(함수)

# 년월일을 입력받는 함수
# 반환값: 년월일을 튜플형태로 반환
def input_date():
    year  = int(input('년도를 입력하시오: '))
    month = int(input('월을 입력하시오: '))
    day   = int(input('일을 입력하시오: '))
    
    return year, month, day

# 윤년인지를 확인하는 함수
# 입력인자: year
# 반환값: True = 윤년, False = 평년
def is_leap(year):
    if year % 400 == 0:                 # 400으로 나뉘는 해: 윤년
        return True
    elif year % 100 == 0:               # 100으로 나뉘는 해: 평년
        return False
    elif year % 4 == 0:                 # 4으로 나뉘는 해  : 윤년
        return True
    else:
        return False

# 요일을 구하는 함수
# 입력인자: year, month, day
# 반환값: 요일
def get_day_name(year, month, day):

    total_days = 0

    year_month_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    day_names = ['일요일','월요일','화요일','수요일','목요일','금요일','토요일']
    
    for item in range(1,year):
        if is_leap(item):
            total_days = total_days + 366
        else:
            total_days = total_days + 365

    for item in range(1,month):
        total_days = total_days + year_month_days[item]

    if month >= 3:
        if is_leap(year):
            total_days = total_days + 1
        else:
            total_days = total_days + 0

    total_days += day
    remainder = total_days % 7
    
    return day_names[remainder]

# main

year, month, day = input_date()
day_name = get_day_name(year, month, day)

print(day_name, '입니다')

### 1.8 예제: 성적 처리 시스템(함수)

# 학생들의 총점을 구하는 함수
def calculate_total(students):
    for student in students:
        student['total'] = student['kor'] + student['eng'] + student['math']

# 학생들의 평균을 구하는 함수
def calculate_average(students):
    for student in students:
        student['avg'] = student['total'] / len(students)

# 학생들의 등수를 구하는 함수
def calculate_order(students):
    temp_students = sorted(students, key = lambda x: x['total'], reverse=True) # 익명함수 활용
    order = 1
    for student in temp_students:
        student['order'] = order
        order = order + 1
    
    return temp_students

# 반 평균을 구하는 함수
def calculate_class_avg(students):
    class_total = 0
    
    for student in students:
        class_total = class_total + student['total']
        
    return class_total / len(students)

# 반의 국어 평균을 구하는 함수
def calculate_kor_avg(students):
    kor_total = 0
    
    for student in students:
        kor_total = kor_total + student['kor']
        
    return kor_total / len(students)

# 반의 영어 평균을 구하는 함수
def calculate_eng_avg(students):
    eng_total = 0
    
    for student in students:
        eng_total = eng_total + student['eng']
        
    return eng_total / len(students)

# 반의 수학 평균을 구하는 함수
def calculate_math_avg(students):
    math_total = 0
    
    for student in students:
        math_total = math_total + student['math']
        
    return math_total / len(students)

# 학생 데이터를 출력하는 함수
def print_student(students):
    for student in students:
        print('번호: {:s}, 이름: {:s}, 국어: {:d}, 영어: {:d}, 수학: {:d}, 총점: {:d}, 평균: {:.2f}, 등수: {:d}'.format(
               student['num'], student['name'],
               student['kor'], student['eng'], student['math'],
               student['total'], student['avg'], student['order'])
             )

# 반의 평균 및 각 과목의 평균을 출력하는 함수
def print_class(class_avg, kor_avg, eng_avg, math_avg):
    print('반 평균  : {:.2f}'.format(class_avg))
    print('국어 평균: {:.2f}'.format(kor_avg)  ) 
    print('영어 평균: {:.2f}'.format(eng_avg)  )
    print('수학 평균: {:.2f}'.format(math_avg) )

# main

students = [
               {'num':'1','name':'김철수','kor':90,'eng':80,'math':85,'total':0,'avg':0.0,'order':0 },
               {'num':'2','name':'박제동','kor':90,'eng':85,'math':90,'total':0,'avg':0.0,'order':0 },
               {'num':'3','name':'홍길동','kor':80,'eng':80,'math':80,'total':0,'avg':0.0,'order':0 }
           ]

class_avg   = 0.0
kor_avg     = 0.0
eng_avg     = 0.0
math_avg    = 0.0

calculate_total(students)                      # 학생의 총점 계산
calculate_average(students)                    # 학생의 평균 계산
students = calculate_order(students)           # 학생의 등수 계산

class_avg   = calculate_class_avg(students)    # 반 평균 계산
kor_avg     = calculate_kor_avg(students)      # 반 국어 평균 계산
eng_avg     = calculate_eng_avg(students)      # 반 영어 평균 계산
math_avg    = calculate_math_avg(students)     # 반 수학 평균 계산


print_student(students)                            # 학생들 데이터 출력
print()
print_class(class_avg, kor_avg, eng_avg, math_avg) # 반 평균, 각 과목 평균 출력


### 1.9 내장 함수(Built-in 함수)

[내장함수 참조 사이트](https://docs.python.org/3.4/library/functions.html)

#### abs()

abs(3)
abs(-3)
abs(-1.2)

#### all()

all([1, 2, 3])

all([1, 2, 3, 0])

#### any()

any([1,2,3,0])

any([0,''])

#### enumerate()

for i, name in enumerate(['body', 'foo', 'bar']):
    print(i, name)

#### isinstance()

isinstance(1, int)

#### len()

len("python")

len([1,2,3])

len((1, 'a'))

#### map()

def two_times(x): 
    return x*2

list(map(two_times, [1, 2, 3, 4]))

#### max()

max([1, 2, 3])

max("python")

#### min()

min([1, 2, 3])
min("python")

#### pow()

pow(2, 3)

#### range()

list(range(5))

list(range(5, 10))

list(range(1, 10, 2))

#### round()

round(4.6)

round(5.678, 2)

#### sorted()

sorted([3, 1, 2])

sorted(['a', 'c', 'b'])

sorted("zero")

sorted((3, 2, 1))

lst = [3, 1, 2]
lst.sort()

#### sum()

sum([1,2,3])

sum((4,5,6))

#### type()

type("abc")

type([ ])

#### zip()

list(zip([1, 2, 3], [4, 5, 6]))

list(zip([1, 2, 3], [4, 5, 6], [7, 8, 9]))

list(zip("abc", "def"))

### 1.10 외장 함수

#### math

import math

math.sin(math.pi)

math.log(math.e)

import math as ma

ma.sin(ma.pi)

ma.log(ma.e)

from math import sin, log, pi, e
sin(pi)

log(e)

from math import *
sin(pi)

log(e)

#### os

import os

os.name

# os.environ

os.environ['PATH']

os.getcwd()

# os.chdir("/home/fdnx")

os.path.join(os.getcwd(), 'tmp')

os.listdir('c:\\')
#os.listdir('/home/fdnx')

os.mkdir( os.path.join(os.getcwd(), 'test'))
os.listdir()

os.remove( os.path.join(os.getcwd(), 'aaa.txt'))
os.listdir()

os.rmdir( os.path.join(os.getcwd(), 'test'))
os.listdir()

os.system("dir")

f = os.popen("dir")
print(f.read())

#### sys

import sys

sys.argv

for item in sys.argv:
    print(item)

sys.copyright

sys.version

sys.path

sys.path.append("/home/fdnx")

# export PYTHONPATH="${PYTHONPATH}:/my/other/path"

#### time

import time

time.time()

time.sleep(1)

input('Enter a number: ')
t1 = time.time()
time.sleep(3)

input('Enter a number again: ')
t2 = time.time()

time_gap = t2 -t1

print('Time gap: ', time_gap)


%time print('aa')
# %timeit print('aa')

%%time
print('aa')
time.sleep(2)
print('bb')
#%%timeit
#print('aa')
#print('bb')

time.gmtime()

time.localtime()

time.asctime(time.localtime())

time.mktime(time.localtime())

#### datetime

import datetime as dtime

dtime.date.today()

dtime.date.fromtimestamp(time.time())

from datetime import datetime

dt = datetime(2011, 10, 29, 20, 30, 21)

dt.day

dt.minute

dt.date()

dt.time()

dt.strftime('%Y-%m-%d %H:%M:%S')

datetime.strptime('20191031092030', '%Y%m%d%H%M%S')

datetime.strptime('20091031', '%Y%m%d')

dt.replace(minute=0, second=0)

dt2 = datetime(2011, 11, 15, 22, 30)
delta = dt2 - dt
delta
type(delta)

dt

dt + delta

[날짜 및 시간 문자열 포멧 참조 사이트](https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior)

<center>날짜 및 시간 지정 | <center> 문자열 의미
:---:|:---
%Y | 앞의 빈자리를 0으로 채우는 4자리 연도 숫자
%m | 앞의 빈자리를 0으로 채우는 2자리 월 숫자
%d | 앞의 빈자리를 0으로 채우는 2자리 일 숫자
%H | 앞의 빈자리를 0으로 채우는 24시간 형식 2자리 시간 숫자
%M | 앞의 빈자리를 0으로 채우는 2자리 분 숫자
%S | 앞의 빈자리를 0으로 채우는 2자리 초 숫자
%A | 영어로 된 요일 문자열
%B | 영어로 된 월 문자열

from dateutil.parser import parse

parse('2016-04-16')

parse("Apr 16, 2016 04:05:32 PM")

parse('6/7/2016')

#### random

import random

random.seed(123)

random.randint(1,10)

random.random()

# lotto_number

import random

def get_lotto_numbers():
    lotto_numbers = []
    
    while True:
        if len(lotto_numbers) == 6:
            break
        
        number = random.randint(1,45)
        if number in lotto_numbers:
            continue
        else:
            lotto_numbers.append(number)
        
    return sorted(lotto_numbers)

lotto_numbers = get_lotto_numbers()
print(lotto_numbers)

### 1.11 라이브러리(패키지)

#### package, module, function

#### mod1.py

```python
def add(a, b): 
    return a+b

def sub(a, b): 
    return a-b
```

#### mod2.py

```python
PI = 3.141592

def add(a, b): 
    return a+b

def sub(a, b): 
    return a-b

if __name__ == "__main__":
    print(add(3, 4))
    print(sub(5, 3))
```

import mod1

mod1.add(2,3)

import mod2

mod2.add(mod2.PI, 3)

#### <참고> sys.path, sys.path.append, PYTHONPATH 환경 변수

#### <참고>pip install package_name

### 1.12 예제: 숫자 맞추기 게임

import random

number = random.randint(1,100)
ans = None

while True:
    print('종료하려면 quit를 입력하세요')
    
    ans = input('1 ~ 100 사이의 숫자를 선택하시오: ')
    
    # 프로그램 종료 체크
    if ans == 'q' or ans == 'quit' or ans == 'exit':
        print('프로그램을 종료합니다.')
        break
        
    # 숫자를 입력했는지 체크
    if not ans.isdecimal():
        print('{}은 잘못된 입력입니다.!!!'.format(ans))
        continue
    
    # 입력값을 숫자로 변환
    num = int(ans)
    
    # 1 ~ 100 사이의 숫자인지 체크
    if num < 1 or num >100:
        print('{}는 1 ~ 100 사이의 숫자가 아닙니다'.format(num))
        continue
    
    # 정답 체크
    if num == number:
        print('축하합니다.')
        print('정답입니다 ==> {}'.format(num))
        print('프로그램을 종료합니다.')
        break
    
    if num > number:
        print('정답은 {}보다 작은수 입니다.'.format(num))
        continue
        
    if num < number:
        print('정답은 {}보다 큰수 입니다.'.format(num))
        continue
        

---

# end of file



# 105. 객체지향 프로그래밍(Class) I

### 1.1 객체와 클래스

- 클래스
- 인스턴스
- 객체(object)

class Car:
    def __init__(self):
        self._speed = 0
        print('자동차가 생성되었습니다.')

    def get_speed(self):
        return self._speed

    def start(self):
        self._speed = 20
        print('자동차가 출발합니다.')

    def accelerate(self):
        self._speed = self._speed + 30
        print('자동차가 가속합니다.')

    def stop(self):
        self._speed = 0
        print('자동차가 정지합니다.')

# main

my_car = Car()
my_car.start()
print('속도 1: ', my_car.get_speed() )
my_car.accelerate()
print('속도 2: ', my_car.get_speed() )
my_car.stop()
print('속도 3: ', my_car.get_speed() )


### 1.2 self

class Car:
    def __init__(self):
        self._speed = 0

    def get_speed(self):
        return self._speed

    def start(self):
        self._speed = 20

    def accelerate(self):
        self._speed = self._speed + 30

    def stop(self):
        self._speed = 0

# main

my_car1 = Car()
my_car2 = Car()

my_car1.start()
my_car2.start()

my_car1.accelerate()

my_car2.accelerate()
my_car2.accelerate()

print('첫번째 자동차 속도: ', my_car1.get_speed() )
print('두번째 자동차 속도: ', my_car2.get_speed() )

my_car1.stop()
my_car2.stop()


class Radio:
    def __init__(self):
        print('라디오가 생성되었습니다.')
        pass

    def turn_on(self):
        print('라디오를 켭니다')
        
    def turn_off(self):
        print('라디오를 끕니다')


class Car:
    def __init__(self):
        self._speed = 0
        print('자동차가 생성되었습니다.')
        self._radio = Radio()

    def get_speed(self):
        return self._speed

    def start(self):
        self._speed = 20
        print('자동차가 출발합니다.')

    def accelerate(self):
        self._speed = self._speed + 30
        print('자동차가 가속합니다.')

    def stop(self):
        self._speed = 0
        print('자동차가 정지합니다.')

    def turn_on_radio(self):
        self._radio.turn_on()

    def turn_off_radio(self):
        self._radio.turn_off()

# main

my_car = Car()
my_car.start()
my_car.turn_on_radio()
my_car.accelerate()
my_car.turn_off_radio()
my_car.stop()


### 1.3 모듈과 클래스

#### vehicle.py

```python

def change_km_to_mile(km):
    return km * 0.621371

def change_mile_to_km(mile):
    return mile * 1.609344


class Car:
    def __init__(self):
        self._speed = 0

    def get_speed(self):
        return self._speed

    def start(self):
        self._speed = 20

    def accelerate(self):
        self._speed = self._speed + 30

    def stop(self):
        self._speed = 0

class Truck:
    def __init__(self):
        self._speed = 0

    def get_speed(self):
        return self._speed

    def start(self):
        self._speed = 10

    def accelerate(self):
        self._speed = self._speed + 20

    def stop(self):
        self._speed = 0


if __name__ == '__main__':
    my_car = Car()
    my_car.start()
    my_car.accelerate()
    print('속도: ', my_car.get_speed() )
    my_car.stop()
```

** name 변수란? **

파이썬의 name 변수는 파이썬이 내부적으로 사용하는 특별한 변수 이름이다. 만약 C:\doit>pythonmod1.py처럼 직접 mod1.py 파일을 실행할 경우 mod1.py의 name 변수에는 main 값이 저장된다. 하지만 파이썬 셸이나 다른 파이썬 모듈에서 mod1을 import 할 경우에는 mod1.py의 name 변수에는 mod1.py의 모듈 이름 값 mod1이 저장된다.


import vehicle
#import vehicle as vh
#from vehicle import *

my_car = vehicle.Car()
#my_car = vh.Car()
#my_car = Car()

my_car.start()
my_car.accelerate()

speed_mile = vehicle.change_km_to_mile(my_car.get_speed())
#speed_mile = vh.change_km_to_mile(my_car.get_speed())
#speed_mile = change_km_to_mile(my_car.get_speed())

print('속도: ', speed_mile, 'mile' )

my_car.stop()


### 1.4 클래스와 데이터 타입

type(1)
type(3.14)
isinstance(1, int)

type('Hello World')
isinstance('Hello World', str)

type([1,2,3])
isinstance([1,2,3], list)

type({'line':1, 'rectangle':2, 'triangle':3})
isinstance({'line':1, 'rectangle':2, 'triangle':3}, dict)


from vehicle import *

my_car = Car()
type(my_car)
isinstance(my_car, Car)

my_truck = Truck()
type(my_truck)
isinstance(my_truck, Truck)


### 1.5 캡슐화와 접근지정

- 캡슐화(encapsulation)
- 상속성(inheritance)
- 다형성(polymorphism)

# access_modifiers1

class Car:
    def __init__(self):
        self.price = 2000
        self._speed = 0
        self.__color = 'red'

my_car = Car()
print(my_car.price)
print(my_car._speed)
print(my_car.__color)


# access_modifiers2

class Car:
    def __init__(self):
        self._price = 0
        self._speed = 0
        self._color = None

    def get_price(self):
        return self._price

    def set_price(self, value):
        self._price = value

    def get_speed(self):
        return self._speed

    def set_speed(self, value):
        self._speed = value

    def get_color(self):
        return self._color

    def set_color(self, value):
        self._color = value

# main

my_car = Car()

my_car.set_price(2000)
my_car.set_speed(20)
my_car.set_color('red')

print('가격: ', my_car.get_price())
print('속도: ', my_car.get_speed())
print('색상: ', my_car.get_color())


### 1.6 property 이용하기

# car_property1

class Car:
    def __init__(self):
        self._price = 0
        self._speed = 0
        self._color = None

    @property
    def price(self):
        return self._price
        
    @price.setter
    def price(self, value):
        self._price = value

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

# main

my_car = Car()

my_car.price = 2000
my_car.speed = 20
my_car.color = 'red'

print('가격: ', my_car.price)
print('속도: ', my_car.speed)
print('색상: ', my_car.color)


# car_property2

class Car:
    def __init__(self):
        self._price = 0
        self._speed = 0
        self._color = None

    def get_price(self):
        return self._price
        
    def set_price(self, value):
        self._price = value
    
    price = property(get_price, set_price)

    def get_speed(self):
        return self._speed

    def set_speed(self, value):
        self._speed = value

    speed = property(get_speed, set_speed)
    
    def get_color(self):
        return self._color

    def set_color(self, value):
        self._color = value
        
    color = property(get_color, set_color)

# main

my_car = Car()

my_car.price = 2000
my_car.speed = 20
my_car.color = 'red'

print('가격: ', my_car.price)
print('속도: ', my_car.speed)
print('색상: ', my_car.color)


### 1.7 예제: 요일 구하기(객체지향)

# 년월일을 입력받는 함수
# 반환값: 년월일을 튜플형태로 반환
def input_date():
    year  = int(input('년도를 입력하시오: '))
    month = int(input('월을 입력하시오: '))
    day   = int(input('일을 입력하시오: '))
    
    return year, month, day


# DayName 클래스

class DayName(object):

    # DayName 클래스 생성자
    def __init__(self, year, month, day):
        self._year  = year
        self._month = month
        self._day   = day
        self._name  = None
        
        self._year_month_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
        self._day_names = ['일요일','월요일','화요일','수요일','목요일','금요일','토요일']
        
        # 요일을 계산한다
        self._calculate_day_name()
        
    # 윤년인지를 체크하는 메서드
    def _is_leap(self, year):
        if year % 400 == 0:                 # 400으로 나뉘는 해: 윤년
            return True
        elif year % 100 == 0:               # 100으로 나뉘는 해: 평년
            return False
        elif year % 4 == 0:                 # 4으로 나뉘는 해  : 윤년
            return True
        else:
            return False
    
    # 요일을 계산하는 메서드
    def _calculate_day_name(self):
    
        total_days = 0
        
        for item in range(1, self._year):
            if self._is_leap(item):
                total_days = total_days + 366
            else:
                total_days = total_days + 365

        for item in range(1, self._month):
            total_days = total_days + self._year_month_days[item]

        if month >= 3:
            if self._is_leap(self._year):
                total_days = total_days + 1

        total_days += self._day
        remainder = total_days % 7
        
        self._name = self._day_names[remainder]
    
    # 요일 이름을 반환하는 메서드
    def get_name(self):
        return self._name
    

# main

year, month, day = input_date()

day_name = DayName(year, month, day)

print( day_name.get_name())


### 1.8 예제: 성적 처리 시스템(객체지향)

# student

class Student(object):

    def __init__(self, num, name, kor, eng, math):
    
        # 학생의 속성
        self._num  = num
        self._name = name
        self._kor  = kor
        self._eng  = eng
        self._math = math
        
        self._total = 0
        self._avg   = 0.0
        self._order = 0
        
        # 총점과 평균을 구한다.
        self._calculate_total()
        self._calculate_avg()
    
    def _calculate_total(self):
        self._total = self._kor + self._eng + self._math
        
    def _calculate_avg(self):
        self._avg = self._total / 3
    
    @property
    def num(self):
        return self._num
        
    @property
    def name(self):
        return self._name
        
    @property
    def kor(self):
        return self._kor
        
    @property
    def eng(self):
        return self._eng
        
    @property
    def math(self):
        return self._math
        
    @property
    def total(self):
        return self._total
        
    @property
    def avg(self):
        return self._avg
        
    @property
    def order(self):
        return self._order
        
    @order.setter
    def order(self, value):
        self._order = value


# grade_system

class StudentGradeSystem(object):

    def __init__(self):
    
        # 학생 객체를 저장할 리스트
        self._students = []
        
        # 반 성적을 저장할 속성
        self._class_avg = 0.0
        self._kor_avg   = 0.0
        self._eng_avg   = 0.0
        self._math_avg  = 0.0

    # 학생 등수를 구한다.
    def _calculate_student_order(self):
        temp_students = sorted(self._students, key = lambda x: x.total, reverse = True)
        
        order = 1
        for student in temp_students:
            student.order = order
            order = order + 1
            
        self._students = temp_students
    
    # 반 평균을 구한다.
    def _calculate_class_avg(self):
        total = 0
        
        for student in self._students:
            total = total + student.total
        
        self._class_avg = total / len(self._students)

    # 국어 평균을 구한다.
    def _calculate_kor_avg(self):
        total = 0
        
        for student in self._students:
            total = total + student.kor
        
        self._kor_avg = total / len(self._students)

    # 영어 평균을 구한다.
    def _calculate_eng_avg(self):
        total = 0
        
        for student in self._students:
            total = total + student.eng
        
        self._eng_avg = total / len(self._students)

    # 수학 평균을 구한다.
    def _calculate_math_avg(self):
        total = 0
        
        for student in self._students:
            total = total + student.math
        
        self._math_avg = total / len(self._students)

    # 반 성적 정보를 구한다.
    def _calculate_class_information(self):
        self._calculate_class_avg()
        self._calculate_kor_avg()
        self._calculate_eng_avg()
        self._calculate_math_avg()

    # 학생 객체를 등록한다.
    def register_student(self, student):
        self._students.append(student)

    # 학생 성적 및 반 성적을 처리한다.
    def process(self):
        self._calculate_student_order()         # 학생 순위를 구한다.
        self._calculate_class_information()     # 반 성적 정보를 구한다.

    # 학생 성적을 출력한다.
    def print_students(self):
        for student in self._students:
            print('번호: {:s}, 이름: {:s}, 국어: {:d}, 영어: {:d}, 수학: {:d}, 총점: {:d}, 평균: {:.2f}, 등수: {:d}'.format(
                   student.num, student.name,
                   student.kor, student.eng, student.math,
                   student.total, student.avg, student.order)
                 )

    # 반 성적 정보를 출력한다.
    def print_class_information(self):
        print('반 평균  : {:.2f}'.format(self._class_avg))
        print('국어 평균: {:.2f}'.format(self._kor_avg)  ) 
        print('영어 평균: {:.2f}'.format(self._eng_avg)  )
        print('수학 평균: {:.2f}'.format(self._math_avg) )


# main

student_grade_system = StudentGradeSystem()

student = Student('1', '김철수', 90, 80, 85)
student_grade_system.register_student(student)

student = Student('2', '박제동', 90, 85, 90)
student_grade_system.register_student(student)

student = Student('3', '홍길동', 80, 80, 80)
student_grade_system.register_student(student)

student_grade_system.process()
student_grade_system.print_students()
print()
student_grade_system.print_class_information()


---

# end of file



# 106. 객체지향 프로그래밍(Class) II

### 1.1 상속

# sportcar_example1

class SportCar(object):
    def __init__(self):
        self._speed = 0

    def get_speed(self):
        return self._speed

    def start(self):
        self._speed = 20

    def accelerate(self):
        self._speed = self._speed + 40
        
    def turbocharge(self):
        self._speed = self._speed + 70

    def stop(self):
        self._speed = 0

# main

my_sportcar = SportCar()
my_sportcar.start()
print('속도 1: ', my_sportcar.get_speed() )
my_sportcar.accelerate()
print('속도 2: ', my_sportcar.get_speed() )
my_sportcar.turbocharge()
print('속도 3: ', my_sportcar.get_speed() )
my_sportcar.stop()


#### <참고> 밑줄 2개로 시작하는 속성이나 메소드는 상속되지 않는다.

# sportcar_example2

class Car(object):
    def __init__(self):
        self._speed = 0

    @property
    def speed(self):
        return self._speed

    def start(self):
        self._speed = 20

    def accelerate(self):
        self._speed = self._speed + 30
        
    def stop(self):
        self._speed = 0
        
class SportCar(Car):
    def __init__(self):
        self._color = 'red'

    def accelerate(self):                  # 오버라이딩(Overriding, 재정의)
        self._speed = self._speed + 40
        
    def turbocharge(self):                 # 메소드 추가
        self._speed = self._speed + 70

    @property
    def color(self):
        return self._color

# main

my_sportcar = SportCar()
print('색상: ', my_sportcar.color )
my_sportcar.start()
print('속도 1: ', my_sportcar.speed )
my_sportcar.accelerate()
print('속도 2: ', my_sportcar.speed )
my_sportcar.turbocharge()
print('속도 3: ', my_sportcar.speed )
my_sportcar.stop()


### 1.2 다형성(Polymorphism)

# airforce

# 인터페이스(Interface)

class AirForce(object):

    def take_off(self):
        pass
    
    def fly(self):
        pass
    
    def attack(self):
        pass
    
    def land(self):
        pass
    

# fighter

# 전투기 클래스
class Fighter(AirForce):

    def __init__(self, weapon_num):
        self._missile_num = weapon_num
    
    def take_off(self):
        print('전투기 발진')
    
    def fly(self):
        print('전투기 목표지로 출격')
    
    def attack(self):
        for i in range(self._missile_num):
            print('미사일 발사')
            self._missile_num -= 1
    
    def land(self):
        print('전투기 착륙')
    

# bomber

# 폭격기 클래스
class Bomber(AirForce):

    def __init__(self, bomb_num):
        self._bomb_num = bomb_num
    
    def take_off(self):
        print('폭격기 발진')
    
    def fly(self):
        print('폭격기 목표지로 출격')
    
    def attack(self):
        for i in range(self._bomb_num):
            print('폭탄 투하')
            self._bomb_num -= 1
    
    def land(self):
        print('폭격기 착륙')
    

# war_game

def war_game(airforce):
    airforce.take_off()
    airforce.fly()
    airforce.attack()
    airforce.land()

# main

f15 = Fighter(3)
war_game(f15)

print()

b29 = Bomber(3)
war_game(b29)


### 1.3 인스턴스 속성과 클래스 속성

# class_variable

class Circle(object):

    PI = 3.14    # 클래스 변수
    
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius

    # 원의 면적을 구한다.
    def get_area(self):
        area = Circle.PI * (self._radius ** 2)    # 클래스명을 이용해서 클래스 변수에 접근
        return round(area, 2)

    # 원의 둘레를 구한다.
    def get_circumference(self):
        circumference = 2 * self.PI * self._radius
        return round(circumference, 2)

# main

print('원주율(클래스 변수): ', Circle.PI)    # 클래스명을 이용해서 클래스 변수에 접근

circle1 = Circle(3)
print('반지름: {}, 면적: {}'.format(circle1.radius, circle1.get_area() ))
print('반지름: {}, 둘레: {}'.format(circle1.radius, circle1.get_circumference() ))

circle2 = Circle(4)
print('반지름: {}, 면적: {}'.format(circle2.radius, circle2.get_area() ))
print('반지름: {}, 둘레: {}'.format(circle2.radius, circle2.get_circumference() ))


### 1.4 인스턴스 메서드, 클래스 메서드, 정적 메서드

#### <참고> 유틸리티 클래스: 클래스 메서드, 정적 메서드들로만 이루어진 클래스

# classmethod_example

class CircleCalculator(object):

    __PI = 3.14    # 외부 접근 허용 불가
    
    # __init__(self) 메서드가 필요없음
    
    # 원의 면적을 구하는 클래스 메서드
    @classmethod
    def calculate_area(cls, radius):       # 인자로 cls를 필요로 한다.
        area = cls.__PI * (radius ** 2)    # cls를 이용해서 클래스 변수에 접근
        return round(area, 2)

    # 원의 둘레를 구하는 클래스 메서드
    @classmethod
    def calculate_circumference(cls, radius):
        circumference = 2 * cls.__PI * radius
        return round(circumference, 2)

# main

print('반지름: 3, 면적: {}'.format(CircleCalculator.calculate_area(3)))
print('반지름: 3, 둘레: {}'.format(CircleCalculator.calculate_circumference(3)))


# staticmethod_example

class CircleCalculator(object):

    # __init__(self) 메서드가 필요없음
    
    # 원의 면적을 구하는 정적 메서드
    @staticmethod
    def calculate_area(radius, pi):        # 호출시 필요한 인자만 사용 (self, cls 필요없음)
        area = pi * (radius ** 2)
        return round(area, 2)

    # 원의 둘레를 구하는 정적 메서드
    @staticmethod
    def calculate_circumference(radius, pi):
        circumference = 2 * pi * radius
        return round(circumference, 2)

# main

print('반지름: 3, 면적: {}'.format(CircleCalculator.calculate_area(3, 3.14)))
print('반지름: 3, 둘레: {}'.format(CircleCalculator.calculate_circumference(3, 3.14)))


### 1.5 예제: 요일 구하기(객체지향 II) - classmethod

# calculate_day_name_classmethod

class DayNameCalculator(object):

    # 클래스 속성 정의
    _year_month_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    _day_names = ['일요일','월요일','화요일','수요일','목요일','금요일','토요일']
        
    # 년월일을 입력받는 클래스 메서드
    @classmethod
    def input_date(cls):
        year  = int(input('년도를 입력하시오: '))
        month = int(input('월을 입력하시오: '))
        day   = int(input('일을 입력하시오: '))
    
        return year, month, day

    # 윤년인지를 체크하는 클래스 메서드
    @classmethod
    def _is_leap(cls, year):
        if year % 400 == 0:                 # 400으로 나뉘는 해: 윤년
            return True
        elif year % 100 == 0:               # 100으로 나뉘는 해: 평년
            return False
        elif year % 4 == 0:                 # 4으로 나뉘는 해  : 윤년
            return True
        else:
            return False
    
    # 요일을 계산하는 클래스 메서드
    @classmethod
    def calculate(cls, year, month, day):
    
        total_days = 0
        
        for item in range(1, year):
            if cls._is_leap(item):
                total_days += 366
            else:
                total_days += 365

        for item in range(1, month):
            total_days += cls._year_month_days[item]

        if month >= 3:
            if cls._is_leap(year):
                total_days += 1

        total_days += day
        
        remainder = total_days % 7
        
        return cls._day_names[remainder]
    

# main

year, month, day = DayNameCalculator.input_date()

day_name = DayNameCalculator.calculate(year, month, day)    # 객체 생성 없이 클래스 메서드 호출

print(day_name)


### 1.6 예제: 요일 구하기(객체지향 II) - staticmethod

# calculate_day_name_staticmethod

class DayNameCalculator(object):

    # 클래스 속성 정의
    _year_month_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    _day_names = ['일요일','월요일','화요일','수요일','목요일','금요일','토요일']
        
    # 년월일을 입력받는 정적 메서드
    @staticmethod
    def input_date():
        year  = int(input('년도를 입력하시오: '))
        month = int(input('월을 입력하시오: '))
        day   = int(input('일을 입력하시오: '))
    
        return year, month, day

    # 윤년인지를 체크하는 정적 메서드
    @staticmethod
    def _is_leap(year):
        if year % 400 == 0:                 # 400으로 나뉘는 해: 윤년
            return True
        elif year % 100 == 0:               # 100으로 나뉘는 해: 평년
            return False
        elif year % 4 == 0:                 # 4으로 나뉘는 해  : 윤년
            return True
        else:
            return False
    
    # 요일을 계산하는 정적 메서드
    @staticmethod
    def calculate(year, month, day):

        total_days = 0
        
        for item in range(1, year):
            if DayNameCalculator._is_leap(item): # 클래스 속성(메소드)사용시: 클래스명으로 접근
                total_days += 366
            else:
                total_days += 365

        for item in range(1, month):
            total_days += DayNameCalculator._year_month_days[item]

        if month >= 3:
            if DayNameCalculator._is_leap(year):
                total_days += 1

        total_days += day
        
        remainder = total_days % 7
        
        return DayNameCalculator._day_names[remainder]
    

# main

year, month, day = DayNameCalculator.input_date()

day_name = DayNameCalculator.calculate(year, month, day)    # 객체 생성 없이 정적 메서드 호출

print(day_name)


#### <참고> 라이브러리 사용

import datetime

date = datetime.date(2014,9,1)
date.strftime('%A')


### 1.7 특별 메서드(매직 메서드)

# car_special_method

class Car(object):

    def __init__(self, name):
        self._name = name

    def __str__(self):
        return 'Type:  Car, Name: ' + self._name
        
my_car = Car('Sonata')
my_car_string = str(my_car)

print(my_car_string)

print(my_car)


# car_special_method2

# 숫자 '+' 연산

class Car(object):

    def __init__(self, price):
        self._price = price

    def __add__(self, val):
        return self._price + val
        
    def __sub__(self, val):
        return self._price - val
        
        
my_car = Car(1000)

print('더하기 연산: ', my_car + 200 )
print('빼기   연산: ', my_car - 200 )


# car_special_method3

# 문자열 '+' 연산

class Car(object):

    def __init__(self, name):
        self._name = name

    def __add__(self, sval):
        return self._name + sval
        
my_car = Car('Sonata')

print('더하기 연산: ', my_car + ' Add string' )


dir(my_car)

#### 연산 관련 메서드
<center>코드|<center>파이썬 호출|<center>설 명
:---|:---|:---
x + y  | x.\_\_add__(y)      |덧셈을 수행한다.
x - y  | x.\_\_sub__(y)      |뺄셈을 수행한다.
x * y  | x.\_\_mul__(y)      |곱셈을 수행한다.
x / y  | x.\_\_truediv__(y)  |나눗셈을 수행한다.
x // y | x.\_\_floordiv__(y) |나눗셈후 버림을 수행한다.
x % y  | x.\_\_mod__(y)      |나머지 연산을 수행한다.
x ** y | x.\_\_pow__(y)      |거듭제곱을 수행한다.

#### 비교 관련 메서드
<center>코드|<center>파이썬 호출|<center>설명
:---|:---|:---
x == y | x.\_\_eq__(y)   |동등성을 평가한다.
x != y | x.\_\_ne__(y)   |동등하지 않음을 평가한다.
x < y  | x.\_\_lt__(y)   |보다 작음을 평가한다.
x <= y | x.\_\_le__(y)   |같거나 작음을 평가한다.
x > y  | x.\_\_gt__(y)   |보다 큼을 평가한다.
x >= y | x.\_\_ge__(y)   |같거나 큼을 평가한다.
if x:  | x.\_\_bool__(y) |x의 불린 값을 구한다.

---

# end of file



# 107. 파일 입출력

### 1.1 파일 생성 및 열기

<center>파일 열기 모드</center>| <center>설 명</center>
----|----
<center>'r'</center> | <div style="text-align: left">읽기 모드로 연다. - Default</div>
<center>'w'</center> | <div style="text-align: left">쓰기 모드로 연다. - 기존 내용 삭제</div>
<center>'a'</center> | <div style="text-align: left">추가 모드로 연다.</div>    
<center>'b'</center> | <div style="text-align: left">이진 모드로 연다.</div>    
<center>'t'</center> | <div style="text-align: left">텍스트 모드로 연다. - Default</div>    

#### write

fp = open('data/hello.txt', 'w')
fp.write('Hello World!')
fp.close()


fp = open('data/text.txt', 'w')
fp.write('{}\n'.format(1))
fp.write('{:.2f}\n'.format(3.1415))
fp.write('Hello World\n')
fp.write('파이썬 문자열입니다\n')
fp.close()
 

#### read line: 한줄씩 읽기

fp = open('data/text.txt', 'r')
line = fp.readline()
print(line.strip())

line = fp.readline()
print(line.strip())

line = fp.readline()
print(line.strip())

line = fp.readline()
print(line.strip())

fp.close()


f = open("text.txt", ’w’)
f.write("Life is too short, you need python")
f.close()

# 파일을 열면 위와 같이 항상 close해 주는 것이 좋다. 하지만 이렇게 파일을 열고 닫는 것을 자동으로 처리할 수 있다면 편리하지 않을까?
# 파이썬의 with문이 바로 이런 역할을 해준다. 다음 예는with문을 사용해서 위 예제를 다시 작성한 모습이다.

with open("foo.txt", "w") as f:
    f.write("Life is too short, you need python")

# with문을 사용하면 with 블록을 벗어나는 순간 열린 파일 객체 f가 자동으로 close되어 편리하다.


#### read lines: 여러줄을 한번에 읽어 리스트에 저장하기

fp = open('data/text.txt', 'r')
lines = fp.readlines()
print(lines)

fp.close()


#### read: 파일 내용 모두 읽기

fp = open('data/text.txt', 'r')
contents = fp.read()
print(contents)

fp.close()


#### 중요 파일 메서드와 속성

<center>메서드| <center>설 명
:---|:---
read([size])        | <div style="text-align: left">파일에서 데이터를 읽어서 문자열로 반환한다. <br> size 인자를 사용해서 몇 바이트를 읽을 것인지 지정할 수 있다.
readline([size])    | <div style="text-align: left">파일에서 한 줄을 읽어 문자열로 반환한다. <br> size 인자를 사용해서 몇 바이트를 읽을 것인지 지정할 수 있다.
readlines([size])   | <div style="text-align: left">파일의 매 줄을 모두 읽어 리스트로 반환한다. <br> size 인자를 사용해서 몇 바이트를 읽을 것인지 지정할 수 있다.
write(str)          | <div style="text-align: left">전달받은 문자열을 파일에 기록한다.
writelines(strings) | <div style="text-align: left">전달받은 문자열 리스트를 파일에 기록한다.
close()             | <div style="text-align: left">파일 핸들을 닫는다.
flush()             | <div style="text-align: left">내부 I/O 버퍼를 디스크로 비운다.
seek(pos)           | <div style="text-align: left">파일 내에서 지정한 위치(정수)로 이동한다.
tell()              | <div style="text-align: left">현재 파일의 위치(정수)를 반환한다.


### 1.2 직렬화 / 역직렬화 (serialize / deserialize)

#### <참고> Protocol Buffers

# pickle_dump

import pickle

fp = open('data/binary.dat', 'wb')

pickle.dump(1, fp)
pickle.dump(3.14, fp)
pickle.dump('Hello World', fp)
pickle.dump('파이썬 문자열입니다', fp)
pickle.dump([1,2,3], fp)
pickle.dump((1,2,3), fp)
pickle.dump({'line':0, 'rectangle':1, 'triangle':2}, fp)

fp.close()


# pickle_load

import pickle

fp = open('data/binary.dat', 'rb')

data = pickle.load(fp)
print(data)

data = pickle.load(fp)
print(data)

data = pickle.load(fp)
print(data)

data = pickle.load(fp)
print(data)

data = pickle.load(fp)
print(data)

data = pickle.load(fp)
print(data)

data = pickle.load(fp)
print(data)


fp.close()


### 1.3 with 문

with open('data/text.txt', 'rt') as fp:
    lines = fp.readlines()
    
    for line in lines:
        print(line.strip())

print('Done!')


### 1.4 예제: 성적 처리 시스템(파일 입출력)

# student.py

class Student(object):

    def __init__(self, num, name, kor, eng, math):
    
        # 학생의 속성
        self._num  = num
        self._name = name
        self._kor  = kor
        self._eng  = eng
        self._math = math
        
        self._total = 0
        self._avg   = 0.0
        self._order = 0
        
        # 총점과 평균을 구한다.
        self._calculate_total()
        self._calculate_avg()
    
    def _calculate_total(self):
        self._total = self._kor + self._eng + self._math
        
    def _calculate_avg(self):
        self._avg = self._total / 3
    
    @property
    def num(self):
        return self._num
        
    @property
    def name(self):
        return self._name
        
    @property
    def kor(self):
        return self._kor
        
    @property
    def eng(self):
        return self._eng
        
    @property
    def math(self):
        return self._math
        
    @property
    def total(self):
        return self._total
        
    @property
    def avg(self):
        return self._avg
        
    @property
    def order(self):
        return self._order
        
    @order.setter
    def order(self, value):
        self._order = value


# grade_system

class StudentGradeSystem(object):

    # StudentGradeSystem 객체의 생성자, 성적 파일을 입력 받는다.
    def __init__(self, score_file):
    
        self._score_file = score_file
    
        # 학생 객체를 저장할 리스트
        self._students = []
        
        # 반 성적을 저장할 속성
        self._class_avg = 0.0
        self._kor_avg   = 0.0
        self._eng_avg   = 0.0
        self._math_avg  = 0.0
        
        self._register_students()
    
    # 성적 파일로부터 학생 데이터를 입력 받는다
    def _register_students(self):
        with open(self._score_file, 'rt', encoding='cp949') as fp:
            lines = fp.readlines()
            
            for line in lines:
                items = (line.strip()).split(',')
                
                num  = items[0]
                name = items[1]
                kor  = int(items[2])
                eng  = int(items[3])
                math = int(items[4])
                
                student = Student(num, name, kor, eng, math)
                self._students.append(student)
                
    # 학생 등수를 구한다.
    def _calculate_student_order(self):
        temp_students = sorted(self._students, key = lambda x: x.total, reverse = True)
        
        order = 1
        for student in temp_students:
            student.order = order
            order = order + 1
            
        self._students = temp_students
    
    # 반 평균을 구한다.
    def _calculate_class_avg(self):
        total = 0
        
        for student in self._students:
            total = total + student.total
        
        self._class_avg = total / len(self._students)

    # 국어 평균을 구한다.
    def _calculate_kor_avg(self):
        total = 0
        
        for student in self._students:
            total = total + student.kor
        
        self._kor_avg = total / len(self._students)

    # 영어 평균을 구한다.
    def _calculate_eng_avg(self):
        total = 0
        
        for student in self._students:
            total = total + student.eng
        
        self._eng_avg = total / len(self._students)

    # 수학 평균을 구한다.
    def _calculate_math_avg(self):
        total = 0
        
        for student in self._students:
            total = total + student.math
        
        self._math_avg = total / len(self._students)

    # 반 성적 정보를 구한다.
    def _calculate_class_information(self):
        self._calculate_class_avg()
        self._calculate_kor_avg()
        self._calculate_eng_avg()
        self._calculate_math_avg()

    # 학생 객체를 등록한다.
    def register_student(self, student):
        self._students.append(student)

    # 학생 성적 및 반 성적을 처리한다.
    def process(self):
        self._calculate_student_order()         # 학생 순위를 구한다.
        self._calculate_class_information()     # 반 성적 정보를 구한다.

    # 처리된 성적 내용을 output_file에 출력한다.
    def output_result(self, output_file):
    
        student_output_format =  '번호: {:2}, 이름: {}, 국어: {}, 영어: {}, 수학: {}, 총점: {}, 평균: {:.2f}, 등수: {}\n'
        
        with open(output_file, 'wt') as fp:
            for student in self._students:
                student_output = student_output_format.format(
                                     student.num, student.name,
                                     student.kor, student.eng, student.math,
                                     student.total, student.avg, student.order)
                fp.write(student_output)
    
            fp.write('\n')
            fp.write('반 평균  : {:.2f}\n'.format(self._class_avg))
            fp.write('국어 평균: {:.2f}\n'.format(self._kor_avg)  ) 
            fp.write('영어 평균: {:.2f}\n'.format(self._eng_avg)  )
            fp.write('수학 평균: {:.2f}\n'.format(self._math_avg) )
            
        print('성적 처리가 끝났습니다.')


# main

# 성적이 기록된 파일을 읽는다.
student_grade_system = StudentGradeSystem('data/score.csv')

# 성적을 처리한다. - 총점, 평균, 등수 등을 계산
student_grade_system.process()

# 처리된 성적을 파일에 기록한다.
student_grade_system.output_result('data/result.csv')


---

# 2. 예외 처리

### 2.1 예외 처리

- 정상종료
- 문법 오류
- 논리 오류
- 예외

# exception_divide1

def divide(m, n):
    return m / n

result = divide(3, 2)
print(result)

result = divide(3, 0)
print(result)


# exception_divide

def divide(m, n):
    try:
        result = m / n
    
    except ZeroDivisionError:
        print('0으로 나눌 수 없습니다.')
    
    except:
        print('ZeroDivisionError 이외의 예외가 발생했습니다.')
    
    else:
        return result
    
    finally:
        print('나눗셈 연산입니다.')

res = divide(3, 2)
print(res)
print()

res = divide(3, 0)
print(res)
print()

res = divide(None, 0)
print(res)


### 2.2 예외 발생시키기

raise TypeError

raise TypeError('타입 오류입니다.')

try:
    raise TypeError('타입 오류입니다.', '이것은 예제입니다.')

except TypeError as e:
    print(e.args)
    

### 2.3 사용자 정의 예외

# age_exception

# 사용자 정의 예외를 정의 했습니다.
class AgeException(Exception):

    def __init__(self, msg):
        self._message = msg
    
# 입력 함수
def input_age():
    age = int(input('나이를 입력해 주세요: '))
        
    if age < 0:
        raise AgeException('나이는 음수가 될 수 없습니다.')
    elif age > 150:
        raise AgeException('150세 이상 살 수 있을까요?')
    else:
        return age
    
# main

try:
    age = input_age()
    
except AgeException as e:
    print(e.args[0])
    
else:
    print('나이는 {}세입니다.'.format(age))
    
    

---

# end of file



# 연습 101 : 파이썬 기초문법

---

money = 10000

if money > 3000:
    print('택시타기')
    print('카카오택시 콜')
else:
    print('버스타기')
    print('걸어가기')

is_there_money = True
if is_there_money:
    print('택시타기')
else:
    print('버스타기')

money = 3000

if money == 3000:
    print('버스타기')
else:
    print('택시타기')

age = 60

if age >= 50:
    print('%d살: 50대 이상!' % age)
elif 40 <= age < 50:
    print('{}살: 40대!'.format(age))
elif 30 <= age < 40:
    print('{}살: 30대!'.format(age))
else:
    print('{}살: 20대나 10대군요?'.format(age))

---

increment = 0

while increment < 10:
    print(increment)
    increment += 1

increment = 10

while increment > 0:
    print(increment)
    increment -= 1

increment = 10

while True:
    print(increment)
    increment -= 1
    if increment == 0:
        break

range(4)

for item in range(4):
    print(item)

for item in range(1, 4):
    print(item)

for item in range(1, 10, 2):
    print(item)

for item in ['냉장고', '세탁기', '선풍기']:
    print(item)

for item in '세탁기':
    print(item)

> **Q. 반복문, 조건문을 이용해서 아래 문제를 풀어보세요.**
> - 놀이공원 입장비를 계산해보세요!
- 입장비는 나이에 따라 다릅니다.
- 사용자로부터 입장할 사람의 수를 무한대로 받을 수 있습니다.
- 사용자가 엔터를 입력하면 끝나고 지금까지 구한 입장비의 총합을 구해서 출력합니다. 

알고리즘 (개발 순서)
1. 먼저 고객의 나이를 받아 변수에 할당해둔다.
2. 해당 변수의 값이 빈값이 아닌동안 아래 내용을 계속 실행한다.
3. 조건문을 이용해서 해당 변수의 값이 연령대에 따라 total_price에 합산될 수 있도록 분기처리한다.
4. 다 합산되었으면 다시 사용자로부터 값을 입력받아 해당 변수에 할당한다.

# 가격
BABY_PRICE  = 1000  # 유아
CHILD_PRICE = 2000  # 청소년
ADULT_PRICE = 3000  # 성인

# 연령대
BABY_LIMIT  = 2   # 2세 이하는 1000원
CHILD_LIMIT = 18  # 18세 이하는 2000원
ADULT_LIMIT = 60  # 60세 이하는 3000원, 60세 이후부터 무료

total_price = 0
age = input('고객의 나이를 입력하세요. (엔터는 종료)')

while age != '':
    age = int(age)
    if age <= BABY_LIMIT:
        total_price += BABY_PRICE
    elif age <= CHILD_LIMIT:
        total_price += CHILD_PRICE
    elif age <= ADULT_LIMIT:
        total_price += ADULT_PRICE
    
    age = input('고객의 나이를 입력하세요. (엔터는 종료)')

print('총 합산 금액은 %d 입니다.' % total_price)


a = 100
b = 10
c = 20
a = a + c
a += 1
a ++
c ++
c#
print(a)

---

name = '파이썬 문자열입니다'

for n in name:
    print(n)

for n in name:
    print(n, end='')

print('aaa', end = ' ')
print('bbb')

for idx, n in enumerate(name):
    print(idx, n)

for k, v in [(1,2),(3,4)]:
    print(k, v)

> **Q. 포수로 시작하는(startswith) 선수를 출력하라**

players = ['투수 임찬국', '투수 류제국 ', '내야수 오진환 ', ' 내야수 가르시아', ' 포수 정상호 ']

for item in players:
    item = item.strip()
    if item.startswith('포수'):
        print(item)

> **Q. 오진환 -> 오지환으로 바꾸어 출력해 보세요.**

players = ['투수 임찬국', '투수 류제국 ', '내야수 오진환 ', ' 내야수 가르시아', ' 포수 정상호 ']

for item in players:
    item = item.strip()
    if item.endswith('오진환'):
        print(item.replace('진', '지'))

for item in players:
    item = item.strip()
    if item in ('내야수 오진환'):
        print(item.replace('진', '지'))

> **Q. 리스트에 있는 원소 중 문자값/Null을 제거한 리스트를 출력하라.**

lst = [1, 2, -8, None, 12, 0, 'NaN']

for item in lst:
    if item is None:
        lst.remove(item)

print(lst)

> **Q. 동물을 잡아서 출력하라!**

> hint) 먼저는 문장 속 단어와 단어를 분리시켜야 합니다. 문자열을 공백을 기준으로 분리시키는 명령어가 무엇인지 생각해보세요.

sentence = "The quick brown fox jumps over the lazy dog"

for item in sentence.split(" "):
    if item in ('fox', 'dog'):
        print('animal:', item)

> **Q. 포지션과 이름을 구분 지어서 튜플로 저장하고 이를 다시 리스트에 저장해보자.**

> **[('투수', '임찬국'), ('투수', '류제국'), ('내야수', '오진환'), ('내야수', '가르시아'), ('포수', '정상호')]**

players = ['투수 임찬국', '투수 류제국', '내야수 오지환', '내야수 가르시아', '포수 정상호']

new_players = []
for player in players:
    p = player.split(" ")
    new_players.append((p[0], p[1]))

print(new_players)

> **Q. 아래 중첩 리스트로 구현된 4층짜리 아파트 호수가 있습니다. **

> 
```python
apart = [[101, 102, 103, 104], [201, 202, 203, 204], [301, 302, 302, 303], [401, 402, 403, 404]]
unsubscribed = [101, 203, 301, 404]
```

apart = [[101, 102, 103, 104], [201, 202, 203, 204], [301, 302, 302, 303], [401, 402, 403, 404]]
unsubscribed = [101, 203, 301, 404]

for floor in apart:
    for house in floor:
        if house in unsubscribed:
            continue
        else:
            print("Newspaper delivery: ", house)

#### 리스트 슬라이싱

a_list = [1, 2, 3, 4, 5]    # list

a_list[2]    == 3           # 특정 위치의 원소 참조. 참조번호는 0번부터 시작.
a_list[-1]   == 5           # 음수인 경우, 뒤에서부터 접근.
a_list[2:4]  == [3,4,5]     # 참조번호 2부터 4까지를 잘라서(slice) 반환함.
a_list[2:]   == [3,4,5]     # 참조번호 2부터 끝까지를 잘라서(slice) 반환함.
a_list[:2]   == [1,2]       # 처음부터 참조번호 2까지를 잘라서(slice) 반환함.
a_list[2::2] == [3,5]       # 참조번호 2부터 끝까지를 자르되, 2칸씩 건너뛰면서 잘라서(slice) 반환함.

> **Q. 아래 리스트에서 포지션을 제외하고 이름만 출력해보세요.**

> **['투수 임찬국', '투수 류제국', '내야수 오지환', '포수 정상호']**

players = ['투수 임찬국', '투수 류제국', '내야수 오지환', '포수 정상호']

for p in players:
    print(p[-3:])

> **Q. 영어 과목의 점수를 출력해보고 미술 과목을 추가해보자.**

marks = {
    '수학': 100,
    '영어': 90,
    '국어': 80
}

marks['영어']

marks.get('영어')

marks['미술'] = 30

marks

print(marks.get('English', 100))

marks['Eng']

marks['미술'] = 70
print(marks)

marks.keys()

marks.values()

marks.items()

> **Q. 포지션별로 선수를 모아보세요.**

players = ['투수 임찬국', '투수 류제국', '내야수 오지환', '내야수 가르시아', '포수 정상호']

{
'투수': ['임찬국', '류제국'],
'내야수': [],
'포수': []
}

by_position = {}

for p in players:
    item = p.split(" ")
    if item[0] not in by_position.keys():
        by_position[item[0]] = []
    by_position[item[0]].append(item[1])

print(by_position)

> **Q. 사용자에게 입력받은 단어의 점수를 Score Card에 따라 표시해보세요.**

alphabets = {
    "D": 5, "E": 4, "F": 1, "G": 8, "H": 1,
    "J": 7, "K": 9, "L": 2, "M": 1, "N": 10, "O": 2, "P": 7, "Q": 6,
    "R": 8, "S": 10, "T": 9, "U": 3, "V": 4, "W": 3 , "X": 6, "Y": 6, "Z": 9
}

word = input('단어를 입력하세요.').upper()

score = 0
for char in word:
    score += alphabets.get(char, 0)

print(score)

> **Q. 3개의 인자값을 받아 평균을 반환하는 함수를 만들어보세요.**

def avg(n1, n2, n3):
    return sum([n1, n2, n3]) / 3

print(avg(1, 2, 3))

> **Q. 미국에 있는 주(State)의 수도는 무엇인지 출력하는 함수 `find_capital`를 만들어보세요.**

# 미국의 주의 수도를 찾는 함수를 만드세요.
STATES_CAPITALS = {
    'Alabama': 'Montgomery',
    'Alaska': 'Juneau',
    'Arizona': 'Phoenix',
    'Wyoming': 'Cheyenne',
}

# Write your code below.
def find_capital(name):
    return STATES_CAPITALS.get(name)

print(find_capital('Alabama'))

---

# end of file



