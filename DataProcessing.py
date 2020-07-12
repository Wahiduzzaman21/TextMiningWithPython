import pandas as pd
import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

def extractuniquetags(df):
    tag_list = []

    for ind in df.index:
        tags = df['tags'][ind]
        separated_tags = tags.split("|")
        for i in separated_tags:
            tag_list.append(i)

    tag_list = list(dict.fromkeys(tag_list))

    return tag_list



def performhotencoding(df,tag_list):
    target = []
    for index in df.index:
        row = []
        for i in tag_list:
            tags = df['tags'][index]
            if (tags.__contains__(i)):
                row.append(1)
            else:
                row.append(0)
        target.append(row)

    dfonehotencoding = pd.DataFrame(target, columns=tag_list)

    return  dfonehotencoding, target


def removepunctuation(df):
    whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
    bangla_fullstop = u"\u0964"
    invalid_chracter = u"\uf06c"
    punctSeq = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
    punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"

    for index in df.index:
        df['content'][index] = whitespace.sub(" ", df['content'][index]).strip()
        df['content'][index] = re.sub(punctSeq, " ", df['content'][index])
        df['content'][index] = re.sub(bangla_fullstop, " ", df['content'][index])
        df['content'][index] = re.sub(invalid_chracter, " ", df['content'][index])
        df['content'][index] = re.sub(punc, " ", df['content'][index])

    return df


def postagging(df):
    from bnlp.bengali_pos import BN_CRF_POS
    bn_pos = BN_CRF_POS()
    model_path = "./model/bn_pos_model.pkl"
    all_content = []

    for index in df.index:
        content = df['content'][index]
        content = bn_pos.pos_tag(model_path, content)
        each_text = []
        for x in content:
            if x[1] == 'NC':
                each_text.append(x[0])
            elif x[1] == 'NP':
                each_text.append(x[0])
        all_content.append(each_text)

    return all_content

def removestopword(all_content):
    with open("./data/stopwords.txt", encoding="utf-8") as file_in:
        stopword = []
        for line in file_in:
            stopword.append(line.rstrip("\n"))

    cleantextwithoutstopword = []
    for x in all_content:
        text = []
        for i in range(len(x)):
            if not stopword.__contains__(x[i]):
                text.append(x[i])

        cleantextwithoutstopword.append(text)

    return cleantextwithoutstopword

def removecharacterenglish(cleantextwithoutstopword):
    with open("./data/specialcharacter.txt", encoding="utf-8") as file_in:
        specialcharacter = []
        for line in file_in:
            specialcharacter.append(line.rstrip("\n"))

    clean_text_v2 = []
    for x in cleantextwithoutstopword:
        text = []
        for i in range(len(x)):
            if not specialcharacter.__contains__(x[i]):
                text.append(x[i])

        clean_text_v2.append(text)

    return clean_text_v2

def joinseparatedtext(clean_text):
    joincontaent = [" ".join(y) for y in clean_text]
    return joincontaent