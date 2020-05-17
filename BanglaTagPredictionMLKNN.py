import pandas as pd

df = pd.read_csv("./data/content/sports/sports_2013.csv",encoding="utf-8-sig")

col_list=[]

for ind in df.index:
    tags = df['tags'][ind]
    separated_tags = tags.split("|")
    for i in separated_tags:
        col_list.append(i)

col_list = list(dict.fromkeys(col_list))
print("Number of unique tags: ",len(col_list))

target =[]
for index in df.index:
    row = []
    for i in col_list:
        tags=df['tags'][index]
        if(tags.__contains__(i)):
            row.append(1)
        else: row.append(0)
    target.append(row)

dfonehotencoding = pd.DataFrame(target, columns = col_list)

# remove punctuation
import re
whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
bangla_fullstop = u"\u0964"
invalid_chracter= u"\uf06c"
punctSeq   = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"

for index in df.index:
    df['content'][index]= whitespace.sub(" ",df['content'][index]).strip()
    df['content'][index] = re.sub(punctSeq, " ", df['content'][index])
    df['content'][index] = re.sub(bangla_fullstop, " ",df['content'][index])
    df['content'][index] = re.sub(invalid_chracter, " ", df['content'][index])
    df['content'][index] = re.sub(punc, " ", df['content'][index])

# pos tagging
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from bnlp.bengali_pos import BN_CRF_POS
bn_pos = BN_CRF_POS()
model_path = "F:/MSCS/Personal/Python/TextMiningWithPython/model/bn_pos_model.pkl"
all_content=[]

for index in df.index:
   content = df['content'][index]
   content = bn_pos.pos_tag(model_path, content)
   each_text=[]
   for x in content:
       if x[1]=='NC':
           each_text.append(x[0])
       elif x[1]=='NP':
           each_text.append(x[0])
   all_content.append(each_text)

# remove stop word
with open("stopwords.txt",encoding="utf-8") as file_in:
    stopword = []
    for line in file_in:
        stopword.append(line.rstrip("\n"))

cleantextwithoutstopword=[]
for x in all_content:
    text=[]
    for i in range(len(x)):
        if not stopword.__contains__(x[i]):
            text.append(x[i])

    cleantextwithoutstopword.append(text)

# remove english special chracter
with open("specialcharacter.txt",encoding="utf-8") as file_in:
    specialcharacter = []
    for line in file_in:
        specialcharacter.append(line.rstrip("\n"))

clean_text_v2=[]
for x in cleantextwithoutstopword:
    text=[]
    for i in range(len(x)):
        if not specialcharacter.__contains__(x[i]):
            text.append(x[i])

    clean_text_v2.append(text)

x_all = clean_text_v2
y_all = target

total_content = x_all
total_target = y_all

total_content =[" ".join(y) for y in total_content]

from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer

df1 = pd.DataFrame(total_content,columns = ['content'])
df2 = pd.DataFrame(dfonehotencoding)
df = df1.join(df2)

print("Number of Articles: ",len(df))

x=df['content'].values
y=df.iloc[:,1:-1].values

cvArticle=CountVectorizer().fit(x)
article=DataFrame(cvArticle.transform(x).todense(),columns=cvArticle.get_feature_names())
x=pd.concat([article],axis=1)

tfidfart=TfidfTransformer().fit(article)
art=DataFrame(tfidfart.transform(article).todense())
x=pd.concat([art],axis=1)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,recall_score,precision_score
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import train_test_split

# Split Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fit into model
classifier = MLkNN(k=44)
classifier.fit(X_train, y_train)

# Predict the result
predictions = classifier.predict(X_test)

# Measure Accuracy, Precision, Recall and F1 Score
print("Accuracy KNN: ",accuracy_score(y_test,predictions))
print("Precision(Micro) KNN: ",precision_score(y_test,predictions,average='micro'))
print("Recall(Micro) KNN: ",recall_score(y_test,predictions,average='micro'))
print("F1 Score(Micro) KNN: ",f1_score(y_test,predictions,average='micro'))