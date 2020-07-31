import pandas as pd
from pandas import DataFrame
import pickle
from DataProcessing import removepunctuation
from DataProcessing import postagging
from DataProcessing import removestopword
from  DataProcessing import removespecialandrenglishchracter
from DataProcessing import joinseparatedtext
from timeit import default_timer as timer

configdf = pd.read_csv("config.csv")



df = pd.read_csv("test.csv", encoding="utf-8-sig")



df = removepunctuation(df)
clean_text_v1 = postagging(df)
clean_text_v2 = removestopword(clean_text_v1)
clean_text_v3 = removespecialandrenglishchracter(clean_text_v2)

total_content = joinseparatedtext(clean_text_v3)
df1 = pd.DataFrame(total_content, columns=['content'])

x = df1['content'].values

vectorizer_model_load_path = "./model/TrainedModel/"+configdf['VectorName'][0]+"/"+configdf['TrainData'][0]+"/"\
                             +str(configdf['Vectorsize'][0])+"/"+configdf['VectorName'][0]+".pkl"
print("Loaded Vectorizer Model:", vectorizer_model_load_path)
countvectorizer_pkl_model = open(vectorizer_model_load_path, 'rb')
countvectorizer_model = pickle.load(countvectorizer_pkl_model)
article = DataFrame(countvectorizer_model.transform(x).todense(), columns=countvectorizer_model.get_feature_names())

vector_transformer_model_load_path = "./model/TrainedModel/TfIdfTransformer/"+configdf['VectorName'][0]+\
                      "/"+configdf['TrainData'][0]+"/"+str(configdf['Vectorsize'][0])+"/"+"TfIdfTransformer.pkl"

print("Loaded Vector Transformer Model:", vector_transformer_model_load_path)
tfidftransformer_pkl_model = open(vector_transformer_model_load_path, 'rb')
tfidftransformer_model = pickle.load(tfidftransformer_pkl_model)
art = DataFrame(tfidftransformer_model.transform(article).todense())

model_load_path = "./model/TrainedModel/MultilabelClassifier/MLKNN/CountVectorizer/Sports/3000/MLKNNCLF.pkl"
print("Loaded Vectorizer Model:", model_load_path)
pkl_model = open(model_load_path, 'rb')
model = pickle.load(pkl_model)

predictions= model.predict(art)

print(predictions)