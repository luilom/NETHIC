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
	pattern1 = re.compile("(\W+|^)\d+(\W+|$)")
	pattern2 = re.compile("(\W+|^)\d+\s+")
	pattern3 = re.compile("(\W+|^)\d+\Z")
	pattern4 = re.compile("\A\d+(\W+|$)")
	pattern5 = re.compile("\s+\d+(\W+|$)")
	doc = pattern1.sub(" ", doc)
	doc = pattern2.sub(" ", doc)
	doc = pattern3.sub(" ", doc)
	doc = pattern4.sub(" ", doc)
	doc = pattern5.sub(" ", doc)
	return (stemmer.stem(w) for w in analyzer(doc))

def __normalizeFeature(dataset):
	dataset = dataset.astype(float)
	i = 0
	j = 0
	for row in dataset:
		j = 0
		size  = row.getnnz()
		for column in row:
			if dataset[i,j] != 0:
				result =  round((round(dataset[i,j],8)/size), 8)
				dataset[i,j] = round(result,8)
			j = j + 1
		i = i + 1
	return dataset

def classify(categoryToLoad,data_test,cutoff,neuralNetwork,dictionary,categories,featureType):

	
    vectorizer = CountVectorizer(decode_error="replace",stop_words='english',vocabulary=dictionary,analyzer=stemmed_words)

    X_testGlobal = vectorizer.fit_transform(data_test.data)
    if featureType == "normalize":
        X_testGlobal = __normalizeFeature(X_testGlobal)
    X_test = normalize(X_testGlobal)

    """load Neural network"""
    
    
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


def recursiveClassification(startCategory,count,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries,categories,featureType):
    if startCategory in categories.keys():
        clf = neuralNetworks["NN_"+startCategory]
        dictionary = dictionaries["dict_"+startCategory]
        result = recursiveClassificationTask(startCategory,count,maxLevel,i,data_test,cutoff,clf,dictionary,categories,featureType)
        if result is not None:
            dictResults[startCategory] = result
            for result in dictResults[startCategory]:
                recursiveClassification(dictResults[startCategory][result].target,count + 1,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries,categories,featureType)
            return dictResults
    return None


def recursiveClassificationTask(startCategory,currLevel,maxLevel,i,data_test,cutoff,clf,dictionary,categories,featureType):
    if int(currLevel) <= int(maxLevel):
        dictScores = dict()
        categories =  classify(startCategory,data_test,cutoff,clf,dictionary,categories,featureType)
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
            path+=str(parents[i].target)+"/"
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
            path+=str(parents[i].target)+"/"
            scorePath = float(scorePath + float(parents[i].score/(i+1)))
        scorePath = float(scorePath/len(parents))
        dictScorePath[path] = scorePath
    return parents[0:len(parents)-1]



def start(neuralNetworks,dictionaries,newData_test,levels,startCategory,metrics,path,featureType):
    categories = json.load(open(path+"/categories.json"))
    cutoff = round(0.8, 2)
    maxIter = 4
    tollerance = 0.7
    currentTollerance = 0
    sorted_results = list()

    while maxIter>0 and currentTollerance<tollerance:
        dictResults = dict()
        dictResults = recursiveClassification(startCategory,0,levels,dictResults,0,newData_test,round(cutoff, 2),neuralNetworks,dictionaries,categories,featureType)
        cutoff = cutoff - round(0.1, 2)
        maxIter = maxIter-1
        results = dict()
        if metrics == "simple":
            computeResult(startCategory,dictResults,results,[])
        elif metrics == "weighed":
            computeResultWeighed(startCategory,dictResults,results,[])
        sorted_results = sorted(results.items(), key=operator.itemgetter(1),reverse=True)
        currentTollerance = 0
        maxToNormalize = sorted_results[0][1]
        for sorted_res in sorted_results:
            """sorted_res è un array che contiene 2 elementi, nella prima posizione c'è il nome della categoria, nella seconda posizione c'è lo score della categoria"""
            currentTollerance=currentTollerance+(sorted_res[1]/maxToNormalize)
    
        currentTollerance = currentTollerance/len(sorted_results)


    return sorted_results

    