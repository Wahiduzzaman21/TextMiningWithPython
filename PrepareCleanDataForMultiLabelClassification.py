import pandas as pd
from DataProcessing import extractuniquetags
from DataProcessing import performhotencoding
from DataProcessing import removepunctuation
from DataProcessing import postagging, postagging_crime
from DataProcessing import removestopword
from  DataProcessing import removespecialandrenglishchracter
from DataProcessing import joinseparatedtext
from timeit import default_timer as timer
import numpy as np


print("Reading Config Started.........")

configdf=pd.read_csv("config.csv")

if configdf['TrainData'][0] == 'Sports':
    df = pd.read_csv("./data/content/sports/AllSportsContent.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'International':
    df = pd.read_csv("./data/content/international/AllInternationalContent.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Bangladesh':
    df = pd.read_csv("./data/content/bangladesh/AllBangladeshContent.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Economy':
    df = pd.read_csv("./data/content/economy/AllEconomyContent.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Entertainment':
    df = pd.read_csv("./data/content/entertainment/AllEntertainmentContent.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Technology':
    df = pd.read_csv("./data/content/technology/AllTechnologyContent.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Crime':
    df = pd.read_csv("./data/content/crime/AllCrimeContent.csv", encoding="utf-8-sig")
else:
    print("Invalid Category")

print("Reading Config Completed.........Target Dataset:"+configdf['TrainData'][0])

print("Data Cleaning Started.........")

# Data Pre-processing
tag_list = extractuniquetags(df)
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
    df.to_csv("./data/content/sports/SportsCleanContentForMultilabelClassification.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'International':
    df.to_csv("./data/content/international/InternationalCleanContentForMultilabelClassification.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Bangladesh':
    df.to_csv("./data/content/bangladesh/BangladeshCleanContentForMultilabelClassification.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Economy':
    df.to_csv("./data/content/economy/EconomyCleanContentForMultilabelClassification.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Entertainment':
    df.to_csv("./data/content/entertainment/EntertainmentCleanContentForMultilabelClassification.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Technology':
    df.to_csv("./data/content/technology/TechnologyCleanContentForMultilabelClassification.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Crime':
    df.to_csv("./data/content/crime/CrimeCleanContentForMultilabelClassification.csv",encoding="utf-8-sig",index=False)
else:
    print("Invalid Category")


print("Writing Clean Data in File Completed................")
print("Total Documents:", df.shape[0])