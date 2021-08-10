### ♧♣ Quiz 01

# 1. 1 이상, 100 이하의 수 중에서 4의 배수 혹은 9의 배수의 총합을 구하시오.

total = 0

for n in range(1, 101):
    if n % 4 == 0 or n % 9 == 0 :
        total += n
print(total)


# 2. 구구단의 7단과 8단을 출력하시오.

num = 0
for i in range(1, 10):
    print('7 *', i, '=', 7*i)
for i in range(1, 10):
    print('8 *', i, '=', 8*i)
   

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

# end
