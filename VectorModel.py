import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pandas import DataFrame
import numpy as np
from itertools import chain


configdf = pd.read_csv("config.csv")

print("Target DataSet:", configdf['TrainData'][0])

if configdf['TrainData'][0] == 'Sports':
    df = pd.read_csv("./data/content/sports/SportsCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'International':
    df = pd.read_csv("./data/content/international/InternationalCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Bangladesh':
    df = pd.read_csv("./data/content/bangladesh/BangladeshCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Economy':
    df = pd.read_csv("./data/content/economy/EconomyCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Entertainment':
    df = pd.read_csv("./data/content/entertainment/EntertainmentCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Technology':
    df = pd.read_csv("./data/content/technology/TechnologyCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Crime':
    df = pd.read_csv("./data/content/crime/CrimeCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
else:
    print("No valid category")



total_content = df['content'].values.tolist()

print("Getting Word Vector list.......................................")

if configdf['TrainData'][0] == 'Sports':
    dfvocablist =  pd.read_csv("./data/vectordata/SportsVector.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'International':
    dfvocablist =  pd.read_csv("./data/vectordata/InternationalVector.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Bangladesh':
    dfvocablist =  pd.read_csv("./data/vectordata/BangladeshVector.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Economy':
    dfvocablist =  pd.read_csv("./data/vectordata/EconomyVector.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Entertainment':
    dfvocablist =  pd.read_csv("./data/vectordata/EntertainmentVector.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Technology':
    dfvocablist =  pd.read_csv("./data/vectordata/TechnologyVector.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Crime':
    dfvocablist =  pd.read_csv("./data/vectordata/CrimeVector.csv", encoding="utf-8-sig")
else:
    print("No valid category")

print("Getting Word Vector list Completed.......................................")


print("Vector Size:", configdf['Vectorsize'][0])

dfvocablist = dfvocablist.head(configdf['Vectorsize'][0])

vocab_list = dfvocablist['Vocabulary'].values.tolist()

vocab_list = list(np.unique(list(chain(vocab_list))))

print(vocab_list)


print("Model Fitting Started...........................................")

df1 = pd.DataFrame(total_content, columns=['content'])
x = df1['content'].values

if configdf['VectorName'][0]=='CountVectorizer':
    model = CountVectorizer(vocabulary=vocab_list).fit(x)
elif configdf['VectorName'][0]=='TfIdfVectorizer':
    model = TfidfVectorizer(vocabulary=vocab_list).fit(x)

print("Model Fitting Completed...........................................")

print("Pickle File Dumping Started...........................................")

model_save_location = "./model/TrainedModel/"+configdf['VectorName'][0]+"/"+configdf['TrainData'][0]+"/"+str(configdf['Vectorsize'][0])+"/"+configdf['VectorName'][0]+".pkl"
print(model_save_location)

pickle.dump(model, open(model_save_location, "wb"))

print("Pickle File Dumping Completed...........................................")

print("Article Shape Checking.....................")

article = DataFrame(model.transform(x).todense(), columns=model.get_feature_names())
print(article.shape)

print("Article Shape Checking Completed.....................")