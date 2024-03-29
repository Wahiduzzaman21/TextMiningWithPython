import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score, precision_recall_fscore_support
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import train_test_split
import pickle
from timeit import default_timer as timer
import math


print("Reading Config Started...............")

configdf = pd.read_csv("config.csv")

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
elif configdf['TrainData'][0] == 'Technology':
    df = pd.read_csv("./data/content/technology/TechnologyCleanContentForMultilabelClassification.csv", encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/TechnologyTag.csv", encoding="utf-8-sig")
else:
    print("Invalid Category")

print("Reading Config Completed...............Target Dataset:"+configdf['TrainData'][0])

print("Word embedding started.................")

# Embedding Text

#count vectorizer
x = df['content'].values

vectorizer_model_load_path = "./model/TrainedModel/"+configdf['VectorName'][0]+"/"+configdf['TrainData'][0]+"/"\
                             +str(configdf['Vectorsize'][0])+"/"+configdf['VectorName'][0]+".pkl"
print("Loaded Vectorizer Model:", vectorizer_model_load_path)
countvectorizer_pkl_model = open(vectorizer_model_load_path, 'rb')
countvectorizer_model = pickle.load(countvectorizer_pkl_model)
article = DataFrame(countvectorizer_model.transform(x).todense(), columns=countvectorizer_model.get_feature_names())
# tfidftransformer

vector_transformer_model_load_path = "./model/TrainedModel/TfIdfTransformer/"+configdf['VectorName'][0]+\
                      "/"+configdf['TrainData'][0]+"/"+str(configdf['Vectorsize'][0])+"/"+"TfIdfTransformer.pkl"

print("Loaded Vector Transformer Model:", vector_transformer_model_load_path)
tfidftransformer_pkl_model = open(vector_transformer_model_load_path, 'rb')
tfidftransformer_model = pickle.load(tfidftransformer_pkl_model)
art = DataFrame(tfidftransformer_model.transform(article).todense())


print("Word embedding completed.................")

# Ready Test Data

x = pd.concat([art], axis=0)
y = df.iloc[:, :-1].values

# Train and Save Model

print("Training Started.............")

# Split Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=0, shuffle=False)

# Fit into model
numofneighbor = int(math.sqrt(X_train.shape[0]))
if numofneighbor%2 ==0:
    numofneighbor+=1
print("Number of K:", numofneighbor)
classifier = MLkNN(k=numofneighbor)

start = timer()
classifier.fit(X_train, y_train)

print("Classifier Best Score:")

print("Time to train the model:", timer()-start)

# Predict the result
predictions = classifier.predict(X_test)


# Measure Accuracy, Precision, Recall and F1 Score
print("Accuracy MLKNN: ", accuracy_score(y_test, predictions))
print("Precision(Micro) MLKNN: ", precision_score(y_test, predictions, average='micro'))
print("Recall(Micro) MLKNN: ", recall_score(y_test, predictions, average='micro'))
print("F1 Score(Micro) MLKNN: ", f1_score(y_test, predictions, average='micro'))

print("Training Completed.............")

print("Start dumping pickle file...........")

mlclf_model_save_location= "./model/TrainedModel/MultilabelClassifier/MLKNN/"+configdf['VectorName'][0]+\
                           "/"+configdf['TrainData'][0]+"/"+str(configdf['Vectorsize'][0])+"/"+"MLKNNCLF.pkl"

pickle.dump(classifier, open(mlclf_model_save_location, "wb"))


print("Pickle Dumping Completed.............")
