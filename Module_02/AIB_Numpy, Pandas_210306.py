# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:56:43 2021

@author: 찬채아빠
"""

#### Day02, 기초 Python Students

string = 'A1B2C3D4E5F6G7H8I9J0' 

string[2:3]

string[1:6:2]

string[:3]

string[::-1]
# s[시작인덱스:종료인덱스:간격]
# ::-1 은 처음부터 끝까지 -1의 간격으로 나열하라는 의미

string[5:1:-1]

string[::-2]

len(string)
# 20

string[len(string)-2::-2]
# (20 - 2)부터 끝까지 -2의 간격으로 나열하라는 의미

string[12:4:-2]

# + 연산자를 활용하는 방법
'abc' + 'ABC'

'abc' * 3

len('abc123')

a = 'abcdefgh'
a[:3] + 'ddd'

a = 'abcdefgh'
a.find('cde')

a='abc_def_abc_ghi_abc'
a.find('abc')

a='abc_def_abc_ghi_abc'
a.find('abc', 0)
a.find('abc', 1)
a.find('abc', 2)
a.find('abc', 3)
a.find('abc', 4)
a.find('abc', 5)
a.find('abc', 6)

print('hello')


# replace 기본 문법
# str.replace(old, new, count)
# count인자를 주지 않게되면 모든 해당하는 문자열을 다 바꿈

a = 'abcdefghabcdefgh'
a.replace('abc', '___')

a.replace('abc', '___', 1)
# abc 를 ____ 로 1번만 바꿔라


a.replace('abc', '___', 2)
# abc 를 ____ 로 2번만 바꿔라

a.replace('abc', '___', 4)
# abc 를 ____ 로 4번만 바꿔라
# abc 가 2개이므로 4번 바꾸라고 해도 결과는 2번 바꾸는 것과 동일함

a = 'a b c d e f g h'
a.split()

a='a,b,c,d,e,f,g,h'
a.split(',')

b='a+b+c+d+e'
b.split(',')

''.join(['a', 'b', 'c'])

'+'.join(['a', 'b', 'c'])

'@'.join('abc')

# 기존 파이썬 포맷팅 기법
'%d %s' % (1, 'test')

# Python 2.6, 3.0에서 추가된 기법 (format method)
'{:.2f} and {}'.format(float(12344.20394), 'string')

# 리스트는 기본적으로 대괄호를 이용한다
a = []
a

a = ['a', 'b', True, 1, 2]
a

a = list('string')
a

# 여기선는 리스트 안에 리스트를 넣어서 구성할 수도 있다. (Nested list)
c = [[1,2,3],[4,5,6],[7,8,9]]
c

L = ['Python', 'is', 'fun']
L[2]

L[-2]

L[2:]

nested_list = [[1,2,3],[4,5,6],[7,8,9]]
nested_list[2]

nested_list[2][0]

# 인덱싱으로 하나의 값을 변경
lst = ['Python', 'is', 'fun']
lst[-1] = 'Not fun'
lst

# 슬라이싱으로 여러개의 값을 변경
lst = ['Python', 'is', 'fun']
lst[1:] = ['is not', 'interesting']
lst

# 만약 바꿀 값의 개수와 추가하는 값의 개수가 다를경우는 어떻게 될까?
lst = [1,2,3,4,5]
lst[2:4] = [1]
lst

# 만약 더 많이 넣는다면 삭제 한 후 리스트를 해당 길이만큼 추가
lst=[1,2,3]
lst[1:] = [1,2,3,4]
lst

lst = [1,2,3]
lst.append('String')
lst

# 원 객체 리스트 자체를 정렬함 (반환값 또한 None)
lst = ['c', 'a', 'A']
lst.sort()
lst

# 정렬된 새로운 리스트를 생성 (반환값은 정렬된 리스트)
lst = ['c', 'a', 'A']
print( sorted(lst) )
print( lst )

# 원래 정렬되는것을 반대로 뒤집고 싶다면?
lst = ['c', 'a', 'A']
lst.sort(reverse=True)
lst

lst=['c', 'b', 'a']
lst.extend(['a', 'b', 'c'])
lst

lst.append(['a', 'b', 'c'])
# append 는 [] 가 추가되고, extend 는 [] 가 추가 안됨

lst = ['a', 'b', 'c']
lst.insert(1, 'd')
lst

lst = ['a', 'b', 'c', 'd']

# [1] 인덱스로 아이템 삭제
del lst[2]
lst

lst = ['a', 'b', 'c', 'd']

del lst[2:4]
lst

lst = ['a', 'b', 'c', 'd']
lst.remove('c')
lst

lst = ['a', 'b', 'c', 'd']
del lst[2]
lst

# 딕셔너리 (Dictionary)¶

# 딕셔너리는 리스트와는 다르게 가변이나 위치에 따라 값이 정의되지 않고 'key' 에 의해 정의됩니다. 말그대로 사전(딕셔너리)같이 어떤 단어가 key에 해당하고 그 뜻이 value에 해당하게됩니다.

D = {}
D

D = {'name': 'Kim', 'age': 20, 'height': 170}
D

D = {'name': ['list']}
D

D = {'name': {'name2': 'kim'}}
D

# 딕셔너리에서 값에 접근하는 방법

# 문자열, 리스트와는 다르게 위치인덱스로 접근하는 것이 아닌 key값을 이용해서 접근
# 형태: dictionary[key]

D = {'name': 'Kim', 'age': 20, 'height': 170}
D['age']

D = {'name': {'name2': 'kim'}}
D['name']['name2']

# 1. 딕셔너리의 항목수(key의 수)를 구하고 싶을 때
D = {'name': 'Kim', 'age': 20, 'height': 170}
len(D)

# 2. 특정 key가 딕셔너리에 있는지 확인하는 방법 (in)
D = {'name': 'Kim', 'age': 20, 'height': 170}
'age' in D

# 문자열, 리스트의 경우에도 멤버쉽 테스트가 가능

# 3. 위에서 정의된 딕셔너리에서 'kim'을 'lee'로 바꾸고 싶다면
D = {'name': {'name2': 'kim'}}
D['name']['name2'] = 'lee'
D

# 4. 딕셔너리에 새로운 키와 값을 추가하고싶을때
D = {'name': 'Kim', 'age': 20, 'height': 170}
D['School'] = 'University'
D

# 5. 딕셔너리의 특정 키 값을 제거 하고 싶을때
D = {'name': 'Kim', 'age': 20, 'height': 170}
del D['name']
D

# 6. 멤버쉽 테스트: 특정 키가 딕셔너리에 있는지 확인하고 싶을때
D = {'name': 'Kim', 'age': 20, 'height': 170}
print( 'age' in D )
print( 'aa' in D)
print( 'age' not in D)

# 1. 모든 key값을 얻고 싶을때
D = {'name': 'Kim', 'age': 20, 'height': 170}
list(D.keys())

# 2. 모든 값을 얻고 싶을때
D = {'name': 'Kim', 'age': 20, 'height': 170}
list(D.values())

# 3. 키와 값을 같이 얻고 싶을때
D = {'name': 'Kim', 'age': 20, 'height': 170}
list(D.items())

a = ()
a

# 콤마가 들어가는 이유는 소괄호는 기본적으로 코드를 감싸는 역할을 함
# (1)을 하게되면 튜플이 생성되는것이 아니라 1을 참조함
# 따라서 파이썬에게는 이것이 튜플이라는 것을 콤마로 알려줘야함
a = (1,)
a

a = (1,2,3)
a

a = None, 'a', 1
a

# 1. 튜플의 합
(1,2,3) + (4,5,6)

# 2. 튜플의 곱
(1,2,3) * 3

file = open('dataset\python_file_example.txt') # txt 파일을 연다

# 반복문을 이용해서 라인 별로 읽기
file = open('dataset\python_file_example.txt') # txt 파일을 연다

for line in file: # 열려있는 파일에서 한 라인별로 읽어온다. 
    print(line, end='')
    
file.close() # 열려있는 외부파일의 연결을 종료한다.

# 다중할당의 경우
# (1)
a = b = c = 0

# (2)
a,b,c = 0,0,0

# (3)
a = 0; b= 0; c = 0;

#(4)
a=0
b=0
c=0

# 1. 문자열을 할당
a,b,c,d = 'PATH'
a,b,c,d

# 2. 리스트를 할당
a,b,c,d = [1,'String',3,4]
a,b,c,d

a,b,c = (1,2,3)
a,b,c

# 4. 부분을 나누어서 할당 (첫번째와 마지막글자 그리고 나머지 글자들로 나누고싶을때)
string = 'PYTHON'
a,b,c = string[0], string[1:-1], string[-1]
a,b,c

# [2] 확장된 시퀀스 언팩킹

# 좌측 할당대상을 임의의 원하는 변수의 길이로 짜르고 싶을 때 의 연산자를 이용해서 할당객체를 원하는 부분만큼만 할당하게 만듬
# \ 연산자에 의해 할당된 변수의 값은 항상 리스트로 반환됨

# 만약, 첫번째 1과 나머지 [2,3,4]를 나누어서 할당하고 싶다면
a,b,c,d = [1,2,3,4]
a, *b = [1,2,3,4]
a,b

*a, b = [1,2,3,4]
a,b

a, *b, c = [1,2,3,4]
a,b,c

a,b,c,d,*e = [1,2,3,4]
e

a = 2
a = a + 1
a

a += 1
a

a -= 1
a

a *= 1
a

a /= 1
a

# [1] 조건문의 형식

a = 3
if a > 0:
    print('Positive integer')
elif a == 0:
    print('a is zero')
else:
    print('invalid input')

# 표현식을 감싸는 ()(괄호)를 활용하는 방법
a = 1
b = 0
if (a == 1 and 
    b == 1):
    print('Yeah')

# [2] 조건 비교시 자주 사용하는 연산자

a = 10; b = -5

if a > 0 and b > 0:
    print(a)
elif a < 0 or b < 0:
    print(b)
else:
    print('?')

#반복적인 작업을 하기 위한 도구로서, while문은 일반적인 루프를 의미하고 for문은 객체의 아이템들을 순차적으로 처리하기 위해서 사용되는 반복문이다.

while 1:
    if condition1:
        break         # 현재 가장 가까운 루프를 탈출
    elif condition2:
        continue     # 현재 가장 가까운 루프의 최상위로 이동
    else:
        pass          # 아무것도 하지 않음 (디버깅시 나중에 채울 코드를 비워두는것보다 pass를 넣어두면 문법에러도 피할 수 있음)

cnt = 0
while 1:
    
    cnt += 1
    if (cnt > 9):
        break               
    elif (cnt < 6):
        continue            
    print(cnt, end= ' ')    

for char in 'thisIsPython':
    print(char, end=' ')
#end=' ' 이걸 해야 가로로 나열됨

for char in 'thisIsPython':
    print(char)
#세로로 실행됨

for each_list in [1, 2, 'string']:
    print(each_list)

T = [(1,2),(3,4)]
for a,b in T:
    print(a,b)

T = [(1,2),(3,4)]
for a,b in T:
    print(a)

T = [(1,2),(3,4),(5,6)]
for a,b,c in T:
    print(a)
#이거는 오류남

#[3] 루프와 관련된 내장함수

#range: 연속되는 정수를 반환
range(0, 5)

list(range(2, 12, 3))

list(range(4, -4, -1))

for i in range(4):
    print( '%s %d' % ('iteration: ', i) )

for i in range(4):
    print('iteration: ', i)

#iteration : 반복

range(0, 5)

list1 = ['Brian','Leo','Kim','Amy']
list2 = [24,29,30,3]

list(zip(list1, list2))

#zip: 인자로 쓰인 객체들을 인덱스의 위치별로 묶어서 튜플로 사용하고 싶은 경우

for (a, b) in zip(list1, list2):
    print(a, b)

for (a, b) in zip(list1, list2):
    print(b)

# 만약 두 객체의 크기가 다를경우에는?
list1 = ['AA', 'BB']
list2 = ['CC', 'DD', 'EE']

for a, b in zip(list1, list2):
    print(a, b)

# enumerate 열거하다

# Enumerate:
for ind, val in enumerate(['a','b','c']):
    print(ind, val)

for ind in enumerate(['a','b','c']):
    print(ind)

#map: 함수와 시퀀스를 인수로 받아 가져온 아이템들을 함수로 호출하여 결과를 얻음

list(map(ord, 'ABCabc'))

#List Comprehension (리스트 컴프리헨션)
#   반복을 이용해 새로운 객체 리스트를 생성할 때 일반적으로 사용하는 여러라인들을 하나의 리스트형태로 구현하는 방식을 의미

# Generate list using multi-lines
lst = []
for i in range(10):
    lst.append(i)
lst

lst = []
for i in range(10):
    lst.append(i)
lst    

# Use only one line
[i for i in range(10)]

#장점1. 코드의 간략화
#장점2. C언어의 속도로 실행되므로, for문보다 더 빠르게 실행

[i for i in range(1, 10) if i % 2 == 0]
# 짝수만 나열하라는 의미
#5 % 2
#Out[]: 1

i = 0
for i in range(1, 10):
    if i % 2 == 0:
        print (i)

[a+b for a in range(1,5) for b in range(2,6)]

a = 0
b = 0
for a in range(1,5):
    for b in range(2,6):
        a+b

#함수 (Function)
#   여러곳에서 여러번 사용되는 로직을 패키지화 시키는 방법 중 하나. 코드 재사용이 용이하고, 또한 복잡한 시스템을 작게 만들어 관리가 용이한 형태로 바꿔준다.

def hello_function():
    print('Hello')
    
hello_function()

def hello_function():
    pass

# 함수에 인자가 여러개이면서, 결과값을 반환할 때는 어떤 형태인가요?
def operator(x, y):
    return x+y+1, x-y+1

 # 위치로 인자를 넣는 방법 (위치인자)
a, b = operator(1, 3)
print(a, b)

# 키워드로 인자를 넣는 방법(키워드인자)
a, b = operator(1, y=3) 
print(a, b)

# 함수의 반환값이 여러개인 경우는 튜플의 형태로 반환됩니다.
result = operator(1, 3)
print(type(result), result)

# **매개변수 기본값 **
# 입력의 매개변수들을 기본값을 넘겨주는건 어떻게 할까요?

def input_func(x=2, y=6, z=9):
    return x + y + z
print( input_func() )
print( input_func(1) )
print( input_func(1, 2) )
print( input_func(1,2,3) )

# 키워드인자를 다 쓸경우, 위치에 관계없습니다.

def operator(x, y):
    return x+y+1, x-y+1

a, b = operator(y=3, x=1)
print(a,b)

# 주의사항1. 기본값이 정의된 인자(키워드인자)는 위치인자 뒤에 와야합니다.

# Case1.
def input_func(x=3, y):
    return x+y+z

# Case2.
def input_func(x, y=3, z):
    return x+y+z

# Case3.
def input_func(x, y, z=3):
    return x+y+z

#Case4.
def input_func(x, y=3, z, a=2):
    return x+y+z

# 주의사항2. 함수의 기본값인자에 할당 시 최초 호출시에만 기본값은 한번만 구해집니다.
# 또한, 인자로 받는 객체가 가변객체(리스트, 딕셔너리 등)일 경우 문제가됩니다.

def test_function(val, L=[]):
    L.append(a)
    return L

# 아래 test_function() 반복 실행하면 [5, 5, 5, 5, 5] 점점 추가됨
    
print( test_function(1) )
print( test_function(2) )
print( test_function(3) )

print(' ')
    
def test_function(val, L=None):
    if L is None:
        L = [] # 새로운 리스트를 재할당
    L.append(a)
    return L

# 아래 test_function() 반복해서 실행해도 결과는 [5] 로 동일함
# 중간에 L = [] 로 새로운 리스트를 재할당 했기 때문임

print( test_function(1) )
print( test_function(2) )
print( test_function(3) )

# 언팩킹: 하나의 객체를 풀어주는 역할; 
# 함수를 호출 시 인자를 풀어주기 때문에 길이에 영향을 받음

def test_func(a, b, c):
    return c, b, a

p = [3,4,5]             
print( test_func(*p) ) # test_func(p[0]. p[1], p[2]) 쓰는 대신
# *p 는 p 리스트 3가지 전체를 의미하는 듯

p = [3,4,5]             
print( test_func(p) ) # 이거는 오류남

print( test_func( *[3,4,5] ) )
print( test_func( *(3,4,5) ) )

print( test_func( [3,4,5] ) ) # *를 안 붙이면 오류남
print( test_func( (3,4,5) ) ) # *를 안 붙이면 오류남

# Error: 반환하는 인자의 수보다 입력인자가 작을경우는 에러
test_func(*(1,2)) 

test_func(*(1,2,3)) # Ok: 개수가 맞음

# 팩킹: 함수가 받을 인자의 개수를 유연하게 정할 수 있습니다.

def test_func(*args):
    return args[0], args[1], args[2]

# 튜플이나, 리스트의 형태가 아니고 숫자를 나열한다면?
test_func(1,2)                  # 입력이 더 적을경우

print( test_func(1,2,3) )       
print( test_func(1,2,3,4) )
print( test_func('a', 'b', 'c') )

# lambda (람다)

# def는 함수를 표현하는 statement이고 이는 리스트 리터럴 혹은 함수인수 내부에서는 사용하기가 힘들다.

# 반변 람다를 이용한 함수 표현식은 작은 인라인 형태로 좀 더 간편한 방법이 된다. 

# 람다 함수 표현은 그 자체로 간단해서 자주 이용될 수 있으나 복잡한 로직이나 많은 것들을 포함할 경우 오히려 가독성을 해치는 경우가 있으므로 그럴 경우는 def를 사용하는 것이 좋다.

f = lambda x: True if x % 2 == 0 else False
f(2)

f = lambda x: True if x % 2 == 0 else False
f(5)

# 리스트내에 함수표현 (def문으로는 불가능)
# 함수정의의 첫번째 방법 

def adder(x, y):
    return x + y
print('def 결과: {}'.format(adder(1,2)) )

f = lambda x, y: x + y
print( 'lambda 결과: {}'.format( f(1,2)) )

# def문과는 달리 인라인 형태로 사용이 가능
# lambda 는 인라인 형태로 사용 가능
func_list = [lambda x:x+1, lambda x: x+2]
for f in func_list:
    print(f(2))

# 모듈 (Module)
    
# 모듈은 우리가 정의하고자 하는것들 (함수, 변수, 클래스 등)을 모아둔 파일(.py)이라고 할 수 있습니다.
    
# 다른 파일 필요한 것들과 필요한 동작들을 정의해두고 따로 관리를 하면 하나의 파일에 전부 다 모아둔 것보다 관리 및 유지보수가 유용합니다.

# import 모듈명
import math
a = math.exp(-1)
print(a)

# from 모듈명 import 클래스/함수/변수
from math import exp # exp 함수를 
a = exp(-1)
print(a)

# 클래스 (Class)

# 클래스는 어떤 추상적인 개념(공장)이고 이 추상적인 개념을 구체적으로 정의하는 것을 인스턴스(구체적인 아이템)라고 합니다.

# 예를들면,축구선수는 어떤 범주를 의미하는 추상적인 개념이죠. 그렇다면 축구선수를 클래스라고 한다면 인스턴스는 어떤것일까요? 축구선수중에서도 박지성, 손흥민 이런식으로 구체화된 것들 인스턴스라고 합니다. 

# 클래스   - 축구선수
# 인스턴스 - 손흥민, 박지성

# 또다른 예로는 게임에서 케릭터를 생성할 때 우리는 능력치가 다 1로 동일한 기본케릭터에서 전사, 마법사 케릭터로 구체화 시킬수 있습니다.

# 회사라는 클래스를 만들고 클래스로부터 사람이라는 인스턴스 생성

# 클래스   - 회사
# 인스턴스 - 사람

# 각 사람들을 구체화하기 위해 사람들의 부서를 결정하는 메서드 생성

class company():
    def setdept(self, department):
        self.dept = department
        
person1 = company() 
person2 = company()
person3 = company()

person1.setdept('Marketing')
person2.setdept('IT')

print( person1.dept )
print( person2.dept )

company.setdept(person3, 'IIT')
print(person3.dept)

# 생성자(__init__)를 이용해서 새로 만들어진 인스턴스 객체를 바로 초기화 시킬수 있습니다.

class company():
    def __init__(self, department): # 
        self.dept = department
        
person1 = company('Marketing')
person2 = company('IT')

print(person1.dept)
print(person2.dept)

# 클래스의 가장 큰 두 가지 특징

# 상속: 부모가 되는 클래스의 변수/함수 등 모든것을 그대로 물려받는것
# 상속에 의한 커스터마이징: 부모 클래스에서 받은 특성을 자식클래스에서 재정의

class parent_class():
    pass
class child_class(parent_class):
    pass

class Company():
    def greeting(self):
        print('Hello, employee')
        
class Employee(Company): # 상속
    def greeting(self):  # 상속 후 greeting을 재정의
        print('Hello, Company')
        
k = Employee()
k.greeting()

#### Quiz

#그렇다면 우리 클래스가 가진 장점이 어떻게 적용되는지 확인해봅시다.
#자 우리가 게임속에서 캐릭터를 만드는 코드를 작성한다고 생각해봅시다.

# 전사, 마법사 두 케릭터를 만든다고 가정해봅시다.
# 세 케릭터 모두 동일한 행동양식(이동, 공격)을 가집니다

# 전사는 기본공격 데미지에 추가데미지가 있습니다. 또한 '돌진' 스킬이 있습니다.

# 마법사는 기본공격 데미지보다 약한 데미지를 가지고 있습니다. 또한 '마법' 스킬이 있습니다.

class character():
    def __init__(self, total_move=0, total_damage=0):
        self.total_damage = total_damage
        self.total_move = total_move
    def move(self, step=1):
        self.total_move += step
    def attack(self, damage=1):
        self.total_damage += damage

char1 = character()
char1.move()
char1.attack()
char1.move()
char1.move()

print('이동3회, 공격1회')
print('총 이동횟수: {}, 총 데미지 {}'.format(char1.total_move, char1.total_damage) )

# 전사 케릭터를 만들어봅시다. 전사는 캐릭터 클래스에서 정의된 (이동, 공격)을 그대로 가지고 있지만, 전사는 데미지가 3입니다. 또한 기본케릭터가 가지고 있지 않는 돌진(dash)라는 스킬이 있습니다. 돌진 스킬은 기본 +3 만큼 이동합니다.

class warrior(character):
    def attack(self, damage=1, add_damage=2):
        character.attack(self, damage+add_damage)
    def dash(self, dash_dist=3): # 돌진 메서드      
        character.move(self, dash_dist)

warrior_instance = warrior()
warrior_instance.move()
warrior_instance.move()
warrior_instance.attack()

print('이동2회, 공격1회')
print('\t총 이동횟수: {}, 총 데미지 {}'.format( warrior_instance.total_move, warrior_instance.total_damage ) )
print(' ')
warrior_instance.dash()
warrior_instance.attack()

print('돌진 1회, 공격 1회')
print('\t총 이동거리: {}, 총 데미지 {}'.format( warrior_instance.total_move, warrior_instance.total_damage ) )

class magician(character):
    def attack(self, damage=1, damage_ratio=0.5):
        character.attack(self, damage*damage_ratio)
    def magic(self): 
        print('마법공격', end='\n')
        
magician_inst = magician()

magician_inst.move()
magician_inst.move()
magician_inst.magic()
magician_inst.attack()
magician_inst.attack()
magician_inst.attack()

print('이동 2회, 마법1회, 공격3회')
print('\t총 이동거리: {}, 총 데미지 {}'.format( magician_inst.total_move, magician_inst.total_damage ) )


#### Day03, Numpy

#Numpy

# Numpy는 수치 프로그래밍을 위한 확장 모듈로서 벡터, 매트릭스 등 수학 및 과학을 위한 고급 도구들을 제공합니다.

# 앞으로 인공지능을 공부하며 사용될 다양한 선형대수 지식들은 이 도구를 이용해서 쉽고 빠르게 계산되어집니다. 이 확장 모듈이 실제로 인공지능에 어떻게 적용되는지는 뒤에 이어질 강의들에서 소개됩니다.

import numpy as np

#### [1] Numpy 어레이 만드는 방법

# Bracket을 이용

np.array([])

np.array([1,2,3])

np.array([1,2,3])
# Ctrl + M 으로 Array 생성할 수 있음

np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=int)

np.array([[1,2,3],
          [4,5,6],
          [7,8,9]])

# Ctrl + M 으로 Array 생성할 수 있음

np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
    
np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float64)

# 2.함수를 이용해서 만드는 방법

# 영으로만 원소를 가지는 행렬 생성
np.zeros((2,2))

# 일로만 원소를 가지는 행렬 생성 
np.ones((3,1))

# 임의의 숫자만을 원소로 가지는 행렬을 생성
np.full((2,4), 10)

# Identity 행렬 생성
np.eye(3) 

# arange: 연속되는 값 생성
np.arange(0, 10, 1)

np.arange(2, 12, 0.5)

# From uniform dsitribution from 0 to 1
np.random.random(3,3)
# 왜 안되는지 모르겠음

# 표준정규분포로부터 얻은 샘플들을 원소로 가지는 행렬 생성
np.random.randn(3,3)

#### [2] 어레이의 접근방법

# 인덱싱

###########


#### Day03, Pandas


    
    
    
    
    
    
    



















