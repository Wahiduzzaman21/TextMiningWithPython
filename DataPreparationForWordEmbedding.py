import pandas as pd

df1 = pd.read_csv("./data/content/sports/AllSportsContent.csv",encoding="utf-8-sig")
df2 = pd.read_csv("./data/content/technology/AllTechnologyContent.csv",encoding="utf-8-sig")
df3 = pd.read_csv("./data/content/entertainment/AllEntertainmentContent.csv",encoding="utf-8-sig")
df4 = pd.read_csv("./data/content/economy/AllEconomyContent.csv",encoding="utf-8-sig")
df5 = pd.read_csv("./data/content/international/AllInternationalContent.csv",encoding="utf-8-sig")
df6 = pd.read_csv("./data/content/bangladesh/AllBangladeshContent.csv",encoding="utf-8-sig")

df1= df1.head(5000)
df2= df2.head(5000)
df3= df3.head(5000)
df4= df4.head(5000)
df5= df5.head(5000)
df6= df6.head(5000)

df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

df.to_csv("./data/vectordata/DataForWordEmbedding.csv",encoding="utf-8-sig",index=False)