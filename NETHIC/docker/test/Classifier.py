import Result
import operator
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from nltk.stem.snowball import EnglishStemmer
import re


def stemmed_words_count(doc):
	stemmer = EnglishStemmer()
	analyzer = CountVectorizer(stop_words='english').build_analyzer()
	pattern = re.compile("\w*\d+\w*")
	doc = pattern.sub(" ", doc)
	return (stemmer.stem(w) for w in analyzer(doc))


def classify(categoryToLoad,data_test,cutoff,neuralNetwork,dictionary,categories):

    vectorizer = CountVectorizer(decode_error="replace",stop_words='english',vocabulary=dictionary,analyzer=stemmed_words_count)

    X_testGlobal = vectorizer.transform(data_test.data)
    X_test = normalize(X_testGlobal)

    """Classifier"""
    pred = neuralNetwork.predict_proba(X_test)

    threshold = (max(pred[0])+min(pred[0]))/2 
    toReturn = []
    
    totalscore = 0
    while totalscore < round(cutoff, 2):
        localscore = pred[0,np.argmax(pred[0,:])]
        category = categories[str(categoryToLoad)][np.argmax(pred[0,:])].lower()
        toReturn.append(Result.Result(category,localscore))
        pred[0,np.argmax(pred[0,:])]=0
        totalscore+=localscore
    return toReturn


def recursiveClassification(startCategory,count,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries,categories):
    if startCategory in categories.keys():
        clf = neuralNetworks["NN_"+startCategory]
        dictionary = dictionaries["dict_"+startCategory]
        result = recursiveClassificationTask(startCategory,count,maxLevel,i,data_test,cutoff,clf,dictionary,categories)
        if result is not None:
            dictResults[startCategory] = result
            for result in dictResults[startCategory]:
                recursiveClassification(dictResults[startCategory][result].target,count + 1,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries,categories)
            return dictResults
    return None


def recursiveClassificationTask(startCategory,currLevel,maxLevel,i,data_test,cutoff,clf,dictionary,categories):
    if int(currLevel) <= int(maxLevel):
        dictScores = dict()
        categories =  classify(startCategory,data_test,cutoff,clf,dictionary,categories)
        if categories is not None:
            for i in range(len(categories)):
                dictScores[str(categories[i].target)] = categories[i]
            return dictScores
    return None   
    


def computeResult(category,dictResults,dictScorePath,parents):
    if category in dictResults.keys():
        for k in dictResults[category].keys():          
            element = dictResults[category][k]
            if parents is None:
                parents = []            
            parents.append(element)
            parents = computeResult(element.target,dictResults,dictScorePath,parents)              
    else:
        path = ""
        scorePath = float(0)
        for i in range(len(parents)):
            path+="/"+str(parents[i].target)
            scorePath = float(scorePath + float(parents[i].score))
        scorePath = float(scorePath/len(parents))
        dictScorePath[path] = scorePath
    return parents[0:len(parents)-1]



def computeResultWeighed(category,dictResults,dictScorePath,parents):
    if category in dictResults.keys():
        for k in dictResults[category].keys():          
            element = dictResults[category][k]
            if parents is None:
                parents = []            
            parents.append(element)
            parents = computeResultWeighed(element.target,dictResults,dictScorePath,parents)              
    else:
        path = ""
        scorePath = float(0)
        for i in range(len(parents)):
            path+="/"+str(parents[i].target)
            scorePath = float(scorePath + float(parents[i].score/(i+1)))
        scorePath = float(scorePath/len(parents))
        dictScorePath[path] = scorePath
    return parents[0:len(parents)-1]



def start(neuralNetworks,dictionaries,newData_test,levels,startCategory,path):
    categories = json.load(open(path+"/categories.json"))
    cutoff = round(0.8, 2)
    maxIter = 4
    tollerance = 0.7
    currentTollerance = 0
    sorted_results = list()

    while maxIter>0 and currentTollerance<tollerance:
        dictResults = dict()
        dictResults = recursiveClassification(startCategory,0,levels,dictResults,0,newData_test,round(cutoff, 2),neuralNetworks,dictionaries,categories)
        cutoff = cutoff - round(0.1, 2)
        maxIter = maxIter-1
        results = dict()
        computeResult(startCategory,dictResults,results,[])
        sorted_results = sorted(results.items(), key=operator.itemgetter(1),reverse=True)
        currentTollerance = 0
        maxToNormalize = sorted_results[0][1]
        for sorted_res in sorted_results:
            """sorted_res è un array che contiene 2 elementi, nella prima posizione c'è il nome della categoria, nella seconda posizione c'è lo score della categoria"""
            currentTollerance=currentTollerance+(sorted_res[1]/maxToNormalize)
    
        currentTollerance = currentTollerance/len(sorted_results)


    return sorted_results

    