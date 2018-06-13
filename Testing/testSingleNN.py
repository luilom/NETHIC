import Classifier
import Result
import pandas as pd
import numpy as np
import sys
import pickle
import operator
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import Bunch
import os
import re
from sklearn.externals import joblib
import json
import time
from nltk.stem.snowball import EnglishStemmer


def stemmed_words_count(doc):
	stemmer = EnglishStemmer()
	analyzer = CountVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))

def stemmed_words_tfidf(doc):
	stemmer = EnglishStemmer()
	analyzer = TfidfVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))



def clearFileName(filename):
    split = filename.split("/")
    pattern1 = re.compile('[0-9]')
    split[-1] = pattern1.sub("", split[-1])
    split[-1]=split[-1][0:-1]
    return split[-1]


#This method return for any leaf a correct parent
def fromLeafToParent(leaf, categories):
    for parent in categories:
        if leaf in categories[parent]:
                return parent
            
    return None
    
def classification(data_test):
    data = []
    with open(data_test.filenames[i], 'rb') as f:
        data.append(f.read())
        data = [d.decode("latin1", "strict") for d in data]


    doc = Bunch(data=data,filenames=data_test.filenames[i],target=data_test.target[i])

    X_test = vectorizer.fit_transform(doc.data)
    pred = model.predict_proba(X_test)

    toReturn = []
    for k in range(0,1):
        category = categories[startCategory][np.argmax(pred[0,:])].lower()
        localscore = pred[0,np.argmax(pred[0,:])]
        pred[0,np.argmax(pred[0,:])] = 0
        toReturn.append(Result.Result(category,localscore))

    return toReturn
    

def startTestNN(levels,startCategory,datasetName,path,featureType,categories,rootCategorization):

    cm =list()
    
    """START"""
    pathNN = str(path)+'/neural_networks_'+str(featureType)
    pathDict = str(path)+'/dictionaries_'+str(featureType)
    categories = json.load(open("categories.json"))
    numCategoryToTake = 1

    #print("Start to load Neural Networks and Dictionaries")

    model = joblib.load(path+'/neural_networks_'+str(featureType)+"/"+"NN_"+startCategory)
    dictionary = joblib.load(path+'/dictionaries_'+str(featureType)+"/"+"dict_"+startCategory+".pkl")
    
    #print("End to load Neural Networks and Dictionaries")

    vectorizer = CountVectorizer(decode_error="replace",stop_words='english',vocabulary=dictionary,analyzer=stemmed_words_count)

    
    """CARICO IL DATASET"""
    data_test = load_files("../datasets/"+datasetName, encoding='latin1')
    globalResults = list()

    
    correctClassifications = 0
    documentTaked = 0
    toReturn = list()

    print(startCategory)
    for i in range(0,len(data_test.data)):
        current = list()
        realCategory = data_test.filenames[i] 
        parentCategory = fromLeafToParent(clearFileName(realCategory),categories)

        if parentCategory == startCategory or rootCategorization == True:
            documentTaked += 1

            data = []
            with open(data_test.filenames[i], 'rb') as f:
                data.append(f.read())
                data = [d.decode("latin1", "strict") for d in data]


            doc = Bunch(data=data,filenames=data_test.filenames[i],target=data_test.target[i])

            X_test = vectorizer.fit_transform(doc.data)
            pred = model.predict_proba(X_test)

            selectedCategory = []
            
            for k in range(0,numCategoryToTake):
                category = categories[startCategory][np.argmax(pred[0,:])].lower()
                localscore = pred[0,np.argmax(pred[0,:])]
                pred[0,np.argmax(pred[0,:])] = 0
                selectedCategory.append(Result.Result(category,localscore))

            for i in range(len(selectedCategory)):
                if rootCategorization == False:
                    if selectedCategory[i].target == clearFileName(realCategory):
                        correctClassifications += 1
                        break
                else:
                    if selectedCategory[i].target == parentCategory:
                        correctClassifications += 1
                        break
            current.append(selectedCategory[i].target)#calcolato
            if rootCategorization:
                if parentCategory == "root":
                    current.append(clearFileName(realCategory.split("/")[-1]))#reale
                else:
                    current.append(parentCategory)#reale
            else:
                current.append(clearFileName(realCategory.split("/")[-1]))#reale

            cm.append(current)

    toReturn.append(startCategory)
    toReturn.append(float(correctClassifications)/float(documentTaked))
    toReturn.append(documentTaked)

    return toReturn,cm
	
        
    

categories = json.load(open("categories.json"))

toSave = list()
for category in categories:
    if category == "root":
        result, cm  = startTestNN(1,category,"datasets_test",".","count",categories,True)
    else:
        result, cm = startTestNN(1,category,"datasets_test",".","count",categories,False)
    toSave.append(result)

    testAccuracyDataFrame = pd.DataFrame(cm, columns=['evaluated','real'])
    testAccuracyDataFrame.to_csv("confusion_matrix/"+str(category)+".csv")

testAccuracyDataFrame = pd.DataFrame(toSave, columns=['category','score','n_samples'])
testAccuracyDataFrame.to_csv("test_accuracy_distribution_singleNN.csv")





