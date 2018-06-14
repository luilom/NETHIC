import Classifier

import pandas
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
import numpy as np

def stemmed_words_count(doc):
	stemmer = EnglishStemmer()
	analyzer = CountVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))

def stemmed_words_tfidf(doc):
	stemmer = EnglishStemmer()
	analyzer = TfidfVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))


def recursiveClassification(startCategory,count,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries,categories):   
    result = recursiveClassificationTask(startCategory,count,maxLevel,i,data_test,cutoff,neuralNetworks,dictionaries,categories)
    if result is not None:
        dictResults[startCategory] = result
        for result in dictResults[startCategory]:
            recursiveClassification(dictResults[startCategory][result].label,count + 1,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries,categories)
        return dictResults
    return None


def recursiveClassificationTask(startCategory,currLevel,maxLevel,i,data_test,cutoff,neuralNetworks,dictionaries,categories):
    if int(currLevel) <= int(maxLevel):
        dictScores = dict()
        categories =  Classifier.classify(startCategory,data_test,cutoff,neuralNetworks,dictionaries,categories)
        if categories is not None:
            for i in range(len(categories)):
                dictScores[str(categories[i].label)] = categories[i]
            return dictScores
    return None   
    


def computeResult(category,dictResults,dictScorePath,parents):
    if category in dictResults.keys():
        for k in dictResults[category].keys():          
            element = dictResults[category][k]
            if parents is None:
                parents = []            
            parents.append(element)
            parents = computeResult(element.label,dictResults,dictScorePath,parents)              
    else:
        path = ""
        scorePath = float(0)
        for i in range(len(parents)):
            path+=str(parents[i].label)+"/"
            scorePath = float(scorePath + (float(parents[i].score)))
        scorePath = float(scorePath/len(parents))
        dictScorePath[path] = scorePath
    return parents[0:len(parents)-1]



def computeResult2(category,dictResults,dictScorePath,parents):
    if category in dictResults.keys():
        for k in dictResults[category].keys():          
            element = dictResults[category][k]
            if parents is None:
                parents = []            
            parents.append(element)
            parents = computeResult2(element.label,dictResults,dictScorePath,parents)              
    else:
        path = ""
        scorePath = float(0)
        for i in range(len(parents)):
            path+=str(parents[i].label)+"/"
            scorePath = float(scorePath + float(parents[i].score/(i+1)))
        scorePath = float(scorePath/len(parents))
        dictScorePath[path] = scorePath
    return parents[0:len(parents)-1]

def clearFileName(filename):
    split = str(data_test.filenames[i]).split("/")
    pattern1 = re.compile('[0-9]')
    split[-1] = pattern1.sub("", split[-1])
    split[-1]=split[-1][0:-1]
    return split[-1]


#This method return for any leaf a correct parent 
def fromLeafToParent(leaf, categories, root):
    for parent in categories:
        if leaf in categories[parent]:
            if parent != root:
                return parent
            else:
                return leaf
    return None


"""START"""

levels = sys.argv[1]
startCategory = sys.argv[2]
pathDataset = sys.argv[3]
pathNNDICT = sys.argv[4]
featureType = sys.argv[5]
taxonomy = sys.argv[6]

pathNN = pathNNDICT+"/NN/"+taxonomy+"/neural_networks_"+str(featureType)
pathDict = pathNNDICT+"/DICT/"+taxonomy+"/dictionaries_"+str(featureType)
categories = json.load(open(str(pathNNDICT+"/"+taxonomy+"_categories.json")))


neuralNetworks = dict()
dictionaries = dict()

print("Start to load Neural Networks and Dictionaries")

"""CARICO LE RETI NEURALI"""
for filename in os.listdir(pathNN):
    neuralNetworks[filename] = joblib.load(pathNN+"/"+filename)

"""CARICO I DIZIONARI"""
for filename in os.listdir(pathDict):
    dictionaries[filename.replace(".pkl","")] = joblib.load(pathDict+"/"+filename)

print("End to load Neural Networks and Dictionaries")




"""CARICO IL DATASET"""
data_test = load_files(pathDataset, encoding='latin1')
globalResults = list()

#CLASSIFICO UN DOCUMENTO ALLA VOLTA con il CUT Scelto
for i in range(0,len(data_test.data)):
    currentClassificationResult = list()
    
    
    cutoff = round(0.8, 2)
    tollerance = 0.7
    data = []
    with open(data_test.filenames[i], 'rb') as f:
        data.append(f.read())
        data = [d.decode("latin1", "strict") for d in data]


    newData_test = Bunch(data=data,filenames=data_test.filenames[i],target=data_test.target[i])
    currentTollerance = 0
    while cutoff>=0.4 and currentTollerance<tollerance:
        currentTollerance = 0
        dictResults = dict()
        startTime = time.time()
        dictResults = recursiveClassification(startCategory,0,levels,dictResults,i,newData_test,round(cutoff, 2),neuralNetworks,dictionaries,categories)
        stopTime = time.time()
        cutoff = cutoff - round(0.1, 2)
        results = dict()
        computeResult(startCategory,dictResults,results,[])
        sorted_results = sorted(results.items(), key=operator.itemgetter(1),reverse=True)
        maxToNormalize = sorted_results[0][1]
        for sorted_res in sorted_results:
            currentTollerance=currentTollerance+(sorted_res[1]/maxToNormalize)
    
        currentTollerance = currentTollerance/len(sorted_results)

    filename = clearFileName(data_test.filenames[i])

    print(sorted_results)
    
    match = 0



    #VERIFICO SE UNA DELLE LABEL CLASSIFICATE COINCIDE CON QUELLA REALE; SE SI METTO CHECK A TRUE
    numCategoryToResults = 0
    if len(sorted_results)>2:
        numCategoryToResults = 3
    else:
        numCategoryToResults = len(sorted_results)
    for j in range(0,numCategoryToResults):
        if str(filename) in sorted_results[j][0]:
                match = 1
                break

    currentClassificationResult.append(data_test.filenames[i].split("/")[-1])

    if len(sorted_results)>=1:
        currentClassificationResult.append(sorted_results[0][0])
    if len(sorted_results)>=2:
        currentClassificationResult.append(sorted_results[1][0])
    if len(sorted_results)>=3:
        currentClassificationResult.append(sorted_results[2][0])
    else:
        for i in range(0,3-len(sorted_results)):
            currentClassificationResult.append("Null")
    



    currentClassificationResult.append(filename)
    currentClassificationResult.append(fromLeafToParent(filename,categories,startCategory))
    currentClassificationResult.append(len(np.unique(data_test.data[i].split(" "))))
    currentClassificationResult.append(round(stopTime-startTime,3))
    currentClassificationResult.append(match)
    
    globalResults.append(currentClassificationResult)
    

if not os.path.exists(taxonomy):
    os.makedirs(taxonomy)
    
dataframeAccuracy = pandas.DataFrame(globalResults, columns=['fileName','label1','label2','label3','realLabel','parent','size','time','match'])
dataframeAccuracy.to_csv(taxonomy+"/classificationResults.csv")

    

    