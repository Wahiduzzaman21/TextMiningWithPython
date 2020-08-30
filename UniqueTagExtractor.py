import pandas as pd
from DataProcessing import extractuniquetags

print("Reading Config Started.........")

configdf = pd.read_csv("config.csv")

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


tag_list = extractuniquetags(df)

dfcol = pd.DataFrame(tag_list,columns=['TagName'])

print("Dumping Tags in File Started.........")

if configdf['TrainData'][0] == 'Sports':
    dfcol.to_csv("./data/uniquetags/SportsTag.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'International':
    dfcol.to_csv("./data/uniquetags/InternationalTag.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Bangladesh':
    dfcol.to_csv("./data/uniquetags/BangladeshTag.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Economy':
    dfcol.to_csv("./data/uniquetags/EconomyTag.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Entertainment':
    dfcol.to_csv("./data/uniquetags/EntertainmentTag.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Technology':
    dfcol.to_csv("./data/uniquetags/TechnologyTag.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Crime':
    dfcol.to_csv("./data/uniquetags/CrimeTag.csv", encoding="utf-8-sig", index=False)
else :
    print("Invalid Category")

print("Dumping Tags in File Completed.........")
print("Total number of tags:", dfcol.shape[0])