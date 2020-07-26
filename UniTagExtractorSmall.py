import pandas as pd
from DataProcessing import extractuniquetags

print("Reading Config Started.........")

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
    print("Invalid Category")

print("Reading Config Completed.........Target Dataset:"+configdf['TrainData'][0])


tag_list = extractuniquetags(df)

dfcol = pd.DataFrame(tag_list,columns=['TagName'])

print("Dumping Tags in File Started.........")

if configdf['TrainData'][0] == 'Sports':
    dfcol.to_csv("./data/uniquetags/SportsTagSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'International':
    dfcol.to_csv("./data/uniquetags/InternationalTagSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Bangladesh':
    dfcol.to_csv("./data/uniquetags/BangladeshTagSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Economy':
    dfcol.to_csv("./data/uniquetags/EconomyTagSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Entertainment':
    dfcol.to_csv("./data/uniquetags/EntertainmentTagSmall.csv",encoding="utf-8-sig",index=False)
elif configdf['TrainData'][0] == 'Technology':
    dfcol.to_csv("./data/uniquetags/TechnologyTagSmall.csv",encoding="utf-8-sig",index=False)
else :
    print("Invalid Category")

print("Dumping Tags in File Completed.........")
print("Total number of tags:", dfcol.shape[0])