from flask import Flask, request
from sklearn.utils import Bunch
import os
import io
import sys
import json
from sklearn.externals import joblib
import Classifier
import json
from flask import Response
from gensim.models.doc2vec import Doc2Vec


app = Flask(__name__)

porta = sys.argv[1]
path = sys.argv[2]

pathNN = path+'/neural_networks'
pathDict = path+'/dictionaries'
pathDoc2Vec = path+'/enwiki_dbow'

neuralNetworks = dict()
dictionaries = dict()

print("Start to load Neural Networks and Dictionaries")

"""CARICO LE RETI NEURALI"""
for filename in os.listdir(pathNN):
    neuralNetworks[filename.replace(".pkl","")] = joblib.load(pathNN+"/"+filename)

"""CARICO I DIZIONARI"""
for filename in os.listdir(pathDict):
    dictionaries[filename.replace(".pkl","")] = joblib.load(pathDict+"/"+filename)

"""CARICO DOC2VEC"""
doc2vec = Doc2Vec.load(pathDoc2Vec+"/doc2vec.bin")

print("End to load Doc2Vec, Neural Networks and Dictionaries")


@app.route("/",methods=['GET','POST'])
def classifier():
    if request.method == 'POST':
        data = request.data
        dataDict = json.loads(data_test)
        text = dataDict.get('text')
    elif request.method == 'GET':
        text = request.args.get('text')

    textToSend = []
    textToSend.append(text)

    """CREO IL BANCH"""
    dataToTest = Bunch(data=textToSend, filenames="",target="")
    text_embedded = doc2vec.infer_vector(textToSend[0].split())

    categoriesSimple = Classifier.start(neuralNetworks,dictionaries,dataToTest,10,"root",path,text_embedded)

    json_simple = json.dumps(categoriesSimple)
    body = "{\"simple\":" + json_simple + "}"
    response = Response(body,mimetype='application/json')
    response.headers.add('content-length', str(len(body))) 
    response.headers.add('Access-Control-Allow-Origin','*')
 
    return response
    
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(porta))