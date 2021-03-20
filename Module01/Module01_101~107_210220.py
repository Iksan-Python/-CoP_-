> ### 빅데이터 분석 및 머신러닝을 위한 파이썬

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
