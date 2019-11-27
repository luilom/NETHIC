import Utility
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

def classify(categoryToLoad, data_test, cutoff, neuralNetworks, dictionaries, categories, text_embedded):
    if categoryToLoad in categories.keys():
        dictionary = dictionaries["dict_" + categoryToLoad]
        vectorizer = CountVectorizer(decode_error="replace", stop_words='english', vocabulary=dictionary, analyzer=stemmed_words_count)
        X_testGlobal = [np.concatenate((text_embedded, np.array(vectorizer.transform(data_test.data).toarray()[0])), axis=None)]

        X_test = normalize(X_testGlobal)
        clf = neuralNetworks["NN_" + categoryToLoad]
        pred = clf.predict_proba(X_test)
        toReturn = []

        totalscore = 0
        while totalscore < round(cutoff, 2):
            localscore = pred[0, np.argmax(pred[0, :])]
            category = categories[str(categoryToLoad)][np.argmax(pred[0, :])].lower()
            toReturn.append(Utility.Result(category, localscore))
            pred[0, np.argmax(pred[0, :])] = 0
            totalscore += localscore

        return toReturn


def recursiveClassification(startCategory,count,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries,categories,text_embedded):
    result = recursiveClassificationTask(startCategory,count,maxLevel,i,data_test,cutoff,neuralNetworks,dictionaries,categories,text_embedded)
    if result is not None:
        dictResults[startCategory] = result
        for result in dictResults[startCategory]:
            recursiveClassification(dictResults[startCategory][result].label,count + 1,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries,categories,text_embedded)
        return dictResults
    return None


def recursiveClassificationTask(startCategory,currLevel,maxLevel,i,data_test,cutoff,neuralNetworks,dictionaries,categories,text_embedded):
    isRoot = False
    if startCategory == "root":
        isRoot = True
    if int(currLevel) <= int(maxLevel):
        dictScores = dict()
        categories_result =  classify(startCategory,data_test,cutoff,neuralNetworks,dictionaries,categories,text_embedded)
        if categories_result is not None:
            for i in range(len(categories_result)):
                dictScores[str(categories_result[i].label)] = categories_result[i]
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
            path+="/"+str(parents[i].label)
            scorePath = float(scorePath + float(parents[i].score))
        scorePath = float(scorePath/len(parents))
        dictScorePath[path] = scorePath
    return parents[0:len(parents)-1]



def start(neuralNetworks,dictionaries,newData_test,levels,startCategory,path,textEmbedded):
    categories = json.load(open(path+"/categories.json"))
    cutoff = round(0.8, 2)
    maxIter = 4
    tollerance = 0.7
    currentTollerance = 0
    sorted_results = list()

    while maxIter>0 and currentTollerance<tollerance:
        dictResults = dict()
        dictResults = recursiveClassification(startCategory,0,levels,dictResults,0,newData_test,round(cutoff, 2),neuralNetworks,dictionaries,categories,textEmbedded)
        cutoff = cutoff - round(0.1, 2)
        maxIter = maxIter-1
        results = dict()
        computeResult(startCategory,dictResults,results,[])
        sorted_results = sorted(results.items(), key=operator.itemgetter(1),reverse=True)
        currentTollerance = 0
        maxToNormalize = sorted_results[0][1]
        for sorted_res in sorted_results:
            currentTollerance=currentTollerance+(sorted_res[1]/maxToNormalize)
    
        currentTollerance = currentTollerance/len(sorted_results)


    return sorted_results

    