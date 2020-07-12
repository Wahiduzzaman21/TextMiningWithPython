import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from DataProcessing import removepunctuation
from DataProcessing import postagging
from DataProcessing import removestopword
from  DataProcessing import removecharacterenglish
from DataProcessing import joinseparatedtext
import pickle

df1 = pd.read_csv("./data/content/sports/AllSportsContent.csv",encoding="utf-8-sig")
df2 = pd.read_csv("./data/content/technology/AllTechnologyContent.csv",encoding="utf-8-sig")
df3 = pd.read_csv("./data/content/entertainment/AllEntertainmentContent.csv",encoding="utf-8-sig")
df4 = pd.read_csv("./data/content/economy/AllEconomyContent.csv",encoding="utf-8-sig")
df5 = pd.read_csv("./data/content/international/AllInternationalContent.csv",encoding="utf-8-sig")
df6 = pd.read_csv("./data/content/bangladesh/AllBangladeshContent.csv",encoding="utf-8-sig")


df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

print("Data Pre Processing Started...................................")

df = removepunctuation(df)
clean_text_v1 = postagging(df)
clean_text_v2 = removestopword(clean_text_v1)
clean_text_v3 = removecharacterenglish(clean_text_v2)
total_content = joinseparatedtext(clean_text_v3)

print("Data Pre Processing Completed...................................")

print("Model Fitting Started...........................................")

df1 = pd.DataFrame(total_content, columns = ['content'])
x = df1['content'].values
model = CountVectorizer().fit(x)

print("Model Fitting Completed...........................................")

print("Pickle File Dumping Started...........................................")

pickle.dump(model, open("./model/CountVectorizer.pkl", "wb"))

print("Pickle File Dumping Completed...........................................")