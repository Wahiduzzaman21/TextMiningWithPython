import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from DataProcessing import removepunctuation
from DataProcessing import postaggingandlemmetization
from DataProcessing import removestopword
from  DataProcessing import removespecialandrenglishchracter
from DataProcessing import joinseparatedtext
import pickle
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
from itertools import chain

df = pd.read_csv("./data/vectordata/DataForWordEmbedding.csv",encoding="utf-8-sig")

print("Data Pre Processing Started...................................")

df = removepunctuation(df)
clean_text_v1 = postaggingandlemmetization(df)
clean_text_v2 = removestopword(clean_text_v1)
clean_text_v3 = removespecialandrenglishchracter(clean_text_v2)
total_content = joinseparatedtext(clean_text_v3)


print("Data Pre Processing Completed...................................")

print("Model Fitting Started...........................................")

df1 = pd.DataFrame(total_content, columns=['content'])
x = df1['content'].values

countvectorizer_pkl_model = open('./model/CountVectorizer.pkl', 'rb')
countvectorizer_model = pickle.load(countvectorizer_pkl_model)

article = DataFrame(countvectorizer_model.transform(x).todense(), columns=countvectorizer_model.get_feature_names())

Tfidfarticle_model = TfidfTransformer().fit(article)

print("Model Fitting Completed...........................................")

print("Pickle File Dumping Started...........................................")

pickle.dump(Tfidfarticle_model, open("./model/TfIdfTransformer.pkl", "wb"))

print("Pickle File Dumping Completed...........................................")

print("Article Shape Checking.....................")
art = DataFrame(Tfidfarticle_model.transform(article).todense())
print(art.shape)

print("Article Shape Checking Completed.....................")

