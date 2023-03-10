import re


customized_set = [
    ('있음', '있음. '),
    ('있음,', '있음.'),
    ('높음 ', '높음.'),
    ('많음', '많음.'),
    ('워함', '워함.'),
    ('행복함', '행복함.'),
    ('관적임', '관적임.'),
    ('줌,', '줌'),
    ('나옴', '나옴.'),
    ('챙겨줌', '챙겨줌.'),
    ('좋다', '좋다.'),
    ('안됨', '안됨.'),
    ('회사임', '회사임.'),
    ('생김', '생김.'),
    ('가능함', '가능함.'),
    ('가능 ', '가능.'),
    ('가능,', '가능.'),
    ('자율 출퇴근', '자율출퇴근'),
    ('다양함 ', '다양함.'),
    ('기대됨 ', '기대됨.'),
    ('는것 ', '는것.'),
    ('은것 ', '은것.'),
    ('한것 ', '한것.'),
    ('할듯 ', '할듯.'),
    ('잘줌 ', '잘줌.'),
    ('잘해줌 ', '잘해줌.'),
    ('싶다 ', '싶다.'),
    ('큼 ', '큼.'),
    ('함 ', '함.'),
    ('줌 ', '줌.'),
    ('없음 ', '없음.'),
    ('잇다 ', '잇다.'),
    ('습니아 ', '습니다.'),
    ('습니다 ', '습니다.'),
    ('무방하다 ', '무방하다.'),
    ('아요 ', '아요.'),
    ('어요 ', '어요.'),
    ('워라벨', '워라밸'),
    ('칠밀도', '친밀도'),
    ('네임벨류', '네임밸류'),
    ('부서바이부서', '부바부'),
    ('부서by부서', '부바부'),
    ('출퇴근 버스', '출퇴근버스'),
    ('셔틀버스', '출퇴근버스'),
    ('셔틀 버스', '출퇴근버스'),
    ('3끼', '삼시세끼'),
    (' .', ''),
    (' - ', '. ')
]

def preprocess_text(sentence):
    for existing, customized in customized_set:
        sentence = sentence.replace(existing, customized)
    sentence = re.sub('[\.]+', '.', sentence)
    return sentence

user_dictionary = """
근무시간\tNNG\t
임금상승\tNNG\t
친밀도\tNNG\t
부바부\tNNG\t
출퇴근버스\tNNG\t
고용안정\tNNG\t
전자결재\tNNG\t

습니아\t습니다/EF\t

자율출퇴근\tNNP\t
자율출퇴근제\t자율출퇴근/NNP\t
자출제\t자율출퇴근/NNP\t

"""

def make_user_dictionary():
    with open('./user_dictionary.txt', 'w', encoding='utf8') as f:
        f.write(user_dictionary)

