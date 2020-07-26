import pandas as pd
from DataProcessing import removepunctuation
from DataProcessing import postagging
from DataProcessing import removestopword
from  DataProcessing import removespecialandrenglishchracter
from DataProcessing import joinseparatedtext
import pickle
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer

configdf = pd.read_csv("config.csv")

print("Target DataSet:", configdf['TrainData'][0])

if configdf['TrainData'][0] == 'Sports':
    df = pd.read_csv("./data/content/sports/SportsCleanContentForMultilabelClassificationSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'International':
    df = pd.read_csv("./data/content/international/InternationalCleanContentForMultilabelClassificationSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Bangladesh':
    df = pd.read_csv("./data/content/bangladesh/BangladeshCleanContentForMultilabelClassificationSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Economy':
    df = pd.read_csv("./data/content/economy/EconomyCleanContentForMultilabelClassificationSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Entertainment':
    df = pd.read_csv("./data/content/entertainment/EntertainmentCleanContentForMultilabelClassificationSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Technology':
    df = pd.read_csv("./data/content/technology/TechnologyCleanContentForMultilabelClassificationSmall.csv", encoding="utf-8-sig")
else:
    print("No valid category")


total_content = df['content'].values.tolist()


print("Model Fitting Started...........................................")

df1 = pd.DataFrame(total_content, columns=['content'])
x = df1['content'].values

model_load_location = "./model/TrainedModel/"+configdf['VectorName'][0]+"/"+configdf['TrainData'][0]+"/"+str(configdf['Vectorsize'][0])+"/"+configdf['VectorName'][0]+"Small.pkl"

print("Load Model Name:", model_load_location)
countvectorizer_pkl_model = open(model_load_location, 'rb')
countvectorizer_model = pickle.load(countvectorizer_pkl_model)


article = DataFrame(countvectorizer_model.transform(x).todense(), columns=countvectorizer_model.get_feature_names())

Tfidfarticle_model = TfidfTransformer().fit(article)

print("Model Fitting Completed...........................................")

print("Pickle File Dumping Started...........................................")

model_save_location = "./model/TrainedModel/TfIdfTransformer/"+configdf['VectorName'][0]+\
                      "/"+configdf['TrainData'][0]+"/"+str(configdf['Vectorsize'][0])+"/"+"TfIdfTransformerSmall.pkl"

pickle.dump(Tfidfarticle_model, open(model_save_location, "wb"))


print("Pickle File Dumping Completed...........................................")

print("Article Shape Checking.....................")
art = DataFrame(Tfidfarticle_model.transform(article).todense())
print(art.shape)
print("Model Save Location:", model_save_location)
print("Article Shape Checking Completed.....................")

