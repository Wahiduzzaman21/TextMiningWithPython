import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

print("Reading Config Started...............")

configdf=pd.read_csv("config.csv")

if configdf['TrainData'][0] == 'Sports':
    df = pd.read_csv("./data/content/sports/SportsCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/SportsTag.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'International':
    df = pd.read_csv("./data/content/international/InternationalCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/InternationalTag.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Bangladesh':
    df = pd.read_csv("./data/content/bangladesh/BangladeshCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/BangladeshTag.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Economy':
    df = pd.read_csv("./data/content/economy/EconomyCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/EconomyTag.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Entertainment':
    df = pd.read_csv("./data/content/entertainment/EntertainmentCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/EntertainmentTag.csv", encoding="utf-8-sig")
else:
    df = pd.read_csv("./data/content/technology/TechnologyCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/TechnologyTag.csv", encoding="utf-8-sig")

print("Reading Config Completed...............Target Dataset:"+configdf['TrainData'][0])

print("Word embedding started.................")

# Embedding Text
x = df['content'].values
countvectorizer_pkl_model = open('./model/CountVectorizer.pkl', 'rb')
countvectorizer_model = pickle.load(countvectorizer_pkl_model)

article = DataFrame(countvectorizer_model.transform(x).todense(), columns=countvectorizer_model.get_feature_names())
Tfidfarticle = TfidfTransformer().fit(article)
art = DataFrame(Tfidfarticle.transform(article).todense())

print("Word embedding completed.................")

# Ready Test Data

x = pd.concat([art],axis=1)
y = df.iloc[:,:-1].values


# Train and Save Model

print("Training Started.............")

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,recall_score,precision_score
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import train_test_split
import pickle

# Split Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fit into model
tag_list = tag_list_df.shape[0]
classifier = MLkNN(k=tag_list)
classifier.fit(X_train, y_train)

# Predict the result
predictions = classifier.predict(X_test)

# Measure Accuracy, Precision, Recall and F1 Score
print("Accuracy MLKNN: ",accuracy_score(y_test,predictions))
print("Precision(Micro) MLKNN: ",precision_score(y_test,predictions,average='micro'))
print("Recall(Micro) MLKNN: ",recall_score(y_test,predictions,average='micro'))
print("F1 Score(Micro) MLKNN: ",f1_score(y_test,predictions,average='micro'))

print("Training Completed.............")

print("Start dumping pickle file...........")

if configdf['TrainData'][0] == 'Sports':
    pickle.dump(classifier, open("./model/TrainedModel/MultilabelClassifier/MLKNN/SportsMultilabelModel.pkl", "wb"))
elif configdf['TrainData'][0] == 'International':
    pickle.dump(classifier, open("./model/TrainedModel/MultilabelClassifier/MLKNN/InternationalMultilabelModel.pkl", "wb"))
elif configdf['TrainData'][0] == 'Bangladesh':
    pickle.dump(classifier, open("./model/TrainedModel/MultilabelClassifier/MLKNN/BangladeshMultilabelModel.pkl", "wb"))
elif configdf['TrainData'][0] == 'Economy':
    pickle.dump(classifier, open("./model/TrainedModel/MultilabelClassifier/MLKNN/EconomyMultilabelModel.pkl", "wb"))
elif configdf['TrainData'][0] == 'Entertainment':
    pickle.dump(classifier, open("./model/TrainedModel/MultilabelClassifier/MLKNN/EntertainmentMultilabelModel.pkl", "wb"))
else:
    pickle.dump(classifier, open("./model/TrainedModel/MultilabelClassifier/MLKNN/TechnologyMultilabelModel.pkl", "wb"))

print("Pickle Dumping Completed.............")