import pandas as pd
import numpy as np

col_list=["title","content","tags","section"]
df = pd.read_csv("./ProthomaloData/content_2013.csv",usecols=col_list,encoding="utf-8-sig")
df = df.loc[df['section'] == "sports"]
df.replace("", np.nan, inplace=True)
df['tags'] = df['tags'].fillna('ফাঁকা')

df = df[df['tags'].str.contains("খেলা")]

print(df.count)


df.to_csv("./data/content/sports/sports_2013.csv",encoding="utf-8-sig",index=False)