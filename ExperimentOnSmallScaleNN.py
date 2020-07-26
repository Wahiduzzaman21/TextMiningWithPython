import pandas as pd
from pandas import DataFrame
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from timeit import default_timer as timer

print("Reading Config Started...............")

configdf = pd.read_csv("config.csv")

if configdf['TrainData'][0] == 'Sports':
    df = pd.read_csv("./data/content/sports/SportsCleanContentForMultilabelClassificationSmall.csv", encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/SportsTagSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'International':
    df = pd.read_csv("./data/content/international/InternationalCleanContentForMultilabelClassificationSmall.csv",
                     encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/InternationalTagSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Bangladesh':
    df = pd.read_csv("./data/content/bangladesh/BangladeshCleanContentForMultilabelClassificationSmall.csv",
                     encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/BangladeshTagSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Economy':
    df = pd.read_csv("./data/content/economy/EconomyCleanContentForMultilabelClassificationSmall.csv", encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/EconomyTagSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Entertainment':
    df = pd.read_csv("./data/content/entertainment/EntertainmentCleanContentForMultilabelClassificationSmall.csv",
                     encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/EntertainmentTagSmall.csv", encoding="utf-8-sig")
elif configdf['TrainData'][0] == 'Technology':
    df = pd.read_csv("./data/content/technology/TechnologyCleanContentForMultilabelClassificationSmall.csv",
                     encoding="utf-8-sig")
    tag_list_df = pd.read_csv("./data/uniquetags/TechnologyTagSmall.csv", encoding="utf-8-sig")
else:
    print("Invalid Category")

print("Reading Config Completed...............Target Dataset:" + configdf['TrainData'][0])

print("Word embedding started.................")

# Embedding Text

# count vectorizer
x = df['content'].values

vectorizer_model_load_path = "./model/TrainedModel/" + configdf['VectorName'][0] + "/" + configdf['TrainData'][0] + "/" \
                             + str(configdf['Vectorsize'][0]) + "/" + configdf['VectorName'][0] + "Small.pkl"
print("Loaded Vectorizer Model:", vectorizer_model_load_path)
countvectorizer_pkl_model = open(vectorizer_model_load_path, 'rb')
countvectorizer_model = pickle.load(countvectorizer_pkl_model)
article = DataFrame(countvectorizer_model.transform(x).todense(), columns=countvectorizer_model.get_feature_names())
# tfidftransformer

vector_transformer_model_load_path = "./model/TrainedModel/TfIdfTransformer/" + configdf['VectorName'][0] + \
                                     "/" + configdf['TrainData'][0] + "/" + str(
    configdf['Vectorsize'][0]) + "/" + "TfIdfTransformerSmall.pkl"

print("Loaded Vector Transformer Model:", vector_transformer_model_load_path)
tfidftransformer_pkl_model = open(vector_transformer_model_load_path, 'rb')
tfidftransformer_model = pickle.load(tfidftransformer_pkl_model)
art = DataFrame(tfidftransformer_model.transform(article).todense())

print("Word embedding completed.................")

# Ready Test Data

x = pd.concat([art], axis=0).to_numpy()
y = df.iloc[:, :-1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


physical_devices = tf.config.list_physical_devices('GPU')

print(physical_devices)

tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.keras.Sequential()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3000, input_shape=(3000,), trainable=False))
model.add(tf.keras.layers.Dense(9000,activation='relu'))
model.add(tf.keras.layers.Dense(9000,activation='relu'))
model.add(tf.keras.layers.Dense(9000,activation='sigmoid'))
model.add(tf.keras.layers.Dense(tag_list_df.count()))
model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=[f1_m,precision_m, recall_m])

# This builds the model for the first time:



start = timer()
# This builds the model for the first time:

model.fit(X_train, y_train, batch_size=128, epochs=500, verbose=1, validation_split=0.1)

print("Time to train the model:", timer()-start)


# This builds the model for the first time:

loss, F1_score, Precision, Recall = model.evaluate(X_test, y_test, batch_size=200)


print("Precision NN", Precision)
print("Recall NN", Recall)
print("F1_score NN", F1_score)

