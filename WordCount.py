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
from DataProcessing import postagging, postagging_crime, postagging_all



print("Reading Config Started...............")

configdf = pd.read_csv("config.csv")

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
else:
    print("Invalid Category")


df = removepunctuation(df)

x = df['content'].values
print(x)

clean_text_v1 = postagging_crime(df)
print(len(clean_text_v1))


word_count=0
for content in clean_text_v1:
    word_count+=len(content)

print(word_count)