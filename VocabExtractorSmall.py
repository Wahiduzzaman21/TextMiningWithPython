import pandas as pd
from DataProcessing import removepunctuation
from DataProcessing import postagging
from DataProcessing import removestopword
from  DataProcessing import removespecialandrenglishchracter
import collections

configdf = pd.read_csv("config.csv")

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
    print("No valid category")


df = removepunctuation(df)
clean_text_v1 = postagging(df)
clean_text_v2 = removestopword(clean_text_v1)
clean_text_v3 = removespecialandrenglishchracter(clean_text_v2)

vocab_list = [j for sub in clean_text_v3 for j in sub]

counts = collections.Counter(vocab_list)
sorted_vocab_list = sorted(vocab_list, key=counts.get, reverse=True)

dfvocab = pd.DataFrame(sorted_vocab_list, columns=['Vocabulary'])

frequency = dfvocab['Vocabulary'].value_counts()

vocab_series = pd.Series(frequency)
print(vocab_series)

vocab_series_top = vocab_series.nlargest(24000)

if configdf['TrainData'][0] == 'Sports':
    vocab_series_top.to_csv("./data/vectordata/SportsVectorSmall.csv", encoding="utf-8-sig", index_label='Vocabulary', header=['Count'])
elif configdf['TrainData'][0] == 'International':
    vocab_series_top.to_csv("./data/vectordata/InternationalVectorSmall.csv", encoding="utf-8-sig", index_label='Vocabulary', header=['Count'])
elif configdf['TrainData'][0] == 'Bangladesh':
    vocab_series_top.to_csv("./data/vectordata/BangladeshVectorSmall.csv", encoding="utf-8-sig", index_label='Vocabulary', header=['Count'])
elif configdf['TrainData'][0] == 'Economy':
    vocab_series_top.to_csv("./data/vectordata/EconomyVectorSmall.csv", encoding="utf-8-sig", index_label='Vocabulary', header=['Count'])
elif configdf['TrainData'][0] == 'Entertainment':
    vocab_series_top.to_csv("./data/vectordata/EntertainmentVectorSmall.csv", encoding="utf-8-sig", index_label='Vocabulary', header=['Count'])
elif configdf['TrainData'][0] == 'Technology':
    vocab_series_top.to_csv("./data/vectordata/TechnologyVectorSmall.csv", encoding="utf-8-sig", index_label='Vocabulary', header=['Count'])
else:
    print("No valid category")

