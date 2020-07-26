import pandas as pd
import numpy as np


col_list = ["title", "content", "tags", "section"]

print("All Content Reading Started...............")

df1 = pd.read_csv("./ProthomaloData/content_2013.csv",usecols=col_list,encoding="utf-8-sig")




# print("Dataframe Shape:", df.shape)

print("All Content Reading Completed...............")

print("Config Reading Started...............")

configdf=pd.read_csv("config.csv")

if configdf['TrainData'][0] == 'Sports':
    sectionname = 'sports'
elif configdf['TrainData'][0] == 'International':
    sectionname = 'international'
elif configdf['TrainData'][0] == 'Bangladesh':
    sectionname = 'bangladesh'
elif configdf['TrainData'][0] == 'Economy':
    sectionname = 'economy'
elif configdf['TrainData'][0] == 'Entertainment':
    sectionname = 'entertainment'
elif configdf['TrainData'][0] == 'Technology':
    sectionname = 'technology'
else:
    print("Invalid Category")


print("Config Reading Completed...............Target Dataset:", sectionname)

df1 = df1.loc[df1['section'] == sectionname]

df = pd.concat([df1], ignore_index=True)

print("Dataframe Shape:", df.shape)

df = df.loc[df['section'] == sectionname]
df.replace("", np.nan, inplace=True)
df.replace(".", np.nan, inplace=True)



df['tags'] = df['tags'].fillna('ফাঁকা')

df['content'] = df['content'].fillna('contentblank')

df = df[~df['tags'].str.contains("ফাঁকা")]

df = df[~df['content'].str.contains("contentblank")]

#df = df[df['tags'].str.contains("খেলা")]


print("Checking any null values", df['content'].isnull().values.any())

print("Checking any null values", df['tags'].isnull().values.any())


print("Dumping Documents in CSV File Started......................")



if configdf['TrainData'][0] == 'Sports':
    df.to_csv("./data/content/sports/AllSportsContentSmall.csv", encoding="utf-8-sig", index=False)
elif configdf['TrainData'][0] == 'International':
    df.to_csv("./data/content/international/AllInternationalContentSmall.csv", encoding="utf-8-sig", index=False)
elif configdf['TrainData'][0] == 'Bangladesh':
    df.to_csv("./data/content/bangladesh/AllBangladeshContentSmall.csv", encoding="utf-8-sig", index=False)
elif configdf['TrainData'][0] == 'Economy':
    df.to_csv("./data/content/economy/AllEconomyContentSmall.csv", encoding="utf-8-sig", index=False)
elif configdf['TrainData'][0] == 'Entertainment':
    df.to_csv("./data/content/entertainment/AllEntertainmentContentSmall.csv", encoding="utf-8-sig", index=False)
elif configdf['TrainData'][0] == 'Technology':
    df.to_csv("./data/content/technology/AllTechnologyContentSmall.csv", encoding="utf-8-sig", index=False)
else:
    print("Invalid Category")




print("Dumping Documents in CSV File Completed......................")

print("Total Number of Documents:", len(df))