import pandas as pd
from DataProcessing import extractuniquetags
from DataProcessing import performhotencoding
from DataProcessing import removepunctuation
from DataProcessing import postagging
from DataProcessing import removestopword
from  DataProcessing import removespecialandrenglishchracter
from DataProcessing import joinseparatedtext
from timeit import default_timer as timer
import numpy as np


print("Reading Config Started.........")

configdf=pd.read_csv("config.csv")

if configdf['TrainData'][0] == 'Sports':
    df = pd.read_csv("./data/content/sports/AllSportsContentSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'International':
    df = pd.read_csv("./data/content/international/AllInternationalContentSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Bangladesh':
    df = pd.read_csv("./data/content/bangladesh/AllBangladeshContentSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Economy':
    df = pd.read_csv("./data/content/economy/AllEconomyContentSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Entertainment':
    df = pd.read_csv("./data/content/entertainment/AllEntertainmentContentSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Technology':
    df = pd.read_csv("./data/content/technology/AllTechnologyContentSmall.csv", encoding="utf-8-sig")
else:
    print("Invalid Category")

print("Reading Config Completed.........Target Dataset:"+configdf['TrainData'][0])

print("Data Cleaning Started.........")

# Data Pre-processing
tag_list = extractuniquetags(df)
print(len(tag_list))
dfonehotencoding = performhotencoding(df, tag_list)

start = timer()

df = removepunctuation(df)
clean_text_v1 = postagging(df)
clean_text_v2 = removestopword(clean_text_v1)
clean_text_v3 = removespecialandrenglishchracter(clean_text_v2)

print("Time to Clean the data:", timer()-start)

total_content = joinseparatedtext(clean_text_v3)

df1 = pd.DataFrame(total_content, columns=['content'])
df2 = pd.DataFrame(dfonehotencoding)
df = df2.join(df1)

df.replace("", np.nan, inplace=True)
df['content'] = df['content'].fillna('contentblank')
df = df[~df['content'].str.contains("contentblank")]
print("Data Cleaning Completed.........")

print("Writing Clean Data in File Started................")

if configdf['TrainData'][0] == 'Sports':
    df.to_csv("./data/content/sports/SportsCleanContentForMultilabelClassificationSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'International':
    df.to_csv("./data/content/international/InternationalCleanContentForMultilabelClassificationSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Bangladesh':
    df.to_csv("./data/content/bangladesh/BangladeshCleanContentForMultilabelClassificationSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Economy':
    df.to_csv("./data/content/economy/EconomyCleanContentForMultilabelClassificationSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Entertainment':
    df.to_csv("./data/content/entertainment/EntertainmentCleanContentForMultilabelClassificationSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Technology':
    df.to_csv("./data/content/technology/TechnologyCleanContentForMultilabelClassificationSmall.csv",encoding="utf-8-sig",index=False)
else:
    print("Invalid Category")


print("Writing Clean Data in File Completed................")
print("Total Documents:", df.shape[0])