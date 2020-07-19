import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from DataProcessing import removepunctuation
from DataProcessing import postagging
from DataProcessing import removestopword
from  DataProcessing import removespecialandrenglishchracter
from DataProcessing import joinseparatedtext
import collections
import numpy as np

configdf = pd.read_csv("config.csv")

print(configdf['TrainData'][0])
print(configdf['VectorName'][0])
print(configdf['Vectorsize'][0])
mlclf_model_save_location= "./model/TrainedModel/MultilabelClassifier/MLKNN/"+configdf['VectorName'][0]+\
                           "/"+configdf['TrainData'][0]+"/"+str(configdf['Vectorsize'][0])+"/"+"MLKNNCLF.pkl"
print(mlclf_model_save_location)
import math

tes = math.sqrt(13781)

print(int(tes))

parameters = {'s': [0.7, 1]}

scorers = {
    'prec': 'precision_micro',
    'rec': 'recall_micro',
    'f1score': 'f1_micro'
}

#grid = GridSearchCV(MLkNN(k=numofneighbor), parameters, scoring=scorers, refit=False, cv=2)
#grid.fit(x, y)
#print("Best Result")
#print(grid.best_params_, grid.best_score_)

#print("All Result")
#print(grid.cv_results_)