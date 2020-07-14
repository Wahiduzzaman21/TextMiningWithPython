import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from DataProcessing import removepunctuation
from DataProcessing import postaggingandlemmetization
from DataProcessing import removestopword
from  DataProcessing import removespecialandrenglishchracter
from DataProcessing import joinseparatedtext
import pickle
from pandas import DataFrame
import numpy as np
from itertools import chain

df = pd.read_csv("./data/vectordata/DataForWordEmbedding.csv",encoding="utf-8-sig")

print("Data Pre Processing Started...................................")

df = removepunctuation(df)
clean_text_v1 = postaggingandlemmetization(df)
clean_text_v2 = removestopword(clean_text_v1)
clean_text_v3 = removespecialandrenglishchracter(clean_text_v2)
total_content = joinseparatedtext(clean_text_v3)

vocab_list = [j for sub in clean_text_v3 for j in sub]

vocab_list = list(np.unique(list(chain(vocab_list))))

print("Data Pre Processing Completed...................................")

print("Model Fitting Started...........................................")

df1 = pd.DataFrame(total_content, columns=['content'])
x = df1['content'].values

model = CountVectorizer(vocabulary=vocab_list).fit(x)

print("Model Fitting Completed...........................................")

print("Pickle File Dumping Started...........................................")

pickle.dump(model, open("./model/CountVectorizer.pkl", "wb"))

print("Pickle File Dumping Completed...........................................")

print("Article Shape Checking.....................")

article = DataFrame(model.transform(x).todense(), columns=model.get_feature_names())
print(article.shape)

print("Article Shape Checking Completed.....................")
