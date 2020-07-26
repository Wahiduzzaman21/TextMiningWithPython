import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pandas import DataFrame
import numpy as np
from itertools import chain
from sklearn.feature_extraction.text import TfidfTransformer


df = pd.read_csv("./data/content/technology/TechnologyCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")

df= df.head(1)

total_content = df['content'].values.tolist()



dfvocablist =  pd.read_csv("./data/vectordata/TechnologyVector.csv", encoding="utf-8-sig")

dfvocablist = dfvocablist.head(3000)
vocab_list = dfvocablist['Vocabulary'].values.tolist()

vocab_list = list(np.unique(list(chain(vocab_list))))

df1 = pd.DataFrame(total_content, columns=['content'])
x = df1['content'].values

model = CountVectorizer(vocabulary=vocab_list).fit(x)
article1 = DataFrame(model.transform(x).todense(), columns=model.get_feature_names())

Tfidfarticle_model = TfidfTransformer().fit(article1)
art1 = DataFrame(Tfidfarticle_model.transform(article1).todense())

#print(article1.values)
print(art1.values)

vectorizer_model_load_path = "./model/TrainedModel/"+"CountVectorizer"+"/"+"Technology"+"/"\
                             +"3000"+"/"+"CountVectorizer"+".pkl"

countvectorizer_pkl_model = open(vectorizer_model_load_path, 'rb')
countvectorizer_model = pickle.load(countvectorizer_pkl_model)
article2 = DataFrame(countvectorizer_model.transform(x).todense(), columns=countvectorizer_model.get_feature_names())



vector_transformer_model_load_path = "./model/TrainedModel/TfIdfTransformer/"+"CountVectorizer"+\
                      "/"+"Technology"+"/"+"3000"+"/"+"TfIdfTransformer.pkl"
tfidftransformer_pkl_model = open(vector_transformer_model_load_path, 'rb')
tfidftransformer_model = pickle.load(tfidftransformer_pkl_model)
art2 = DataFrame(tfidftransformer_model.transform(article2).todense())
print(art2.values)


