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

app = Flask(__name__)

porta = sys.argv[1]
path = sys.argv[2]

pathNN = path+'/neural_networks'
pathDict = path+'/dictionaries'
neuralNetworks = dict()
dictionaries = dict()

print("Start to load Neural Networks and Dictionaries")

"""CARICO LE RETI NEURALI"""
for filename in os.listdir(pathNN):
    neuralNetworks[filename.replace(".pkl","")] = joblib.load(pathNN+"/"+filename)

print(neuralNetworks.keys())
"""CARICO I DIZIONARI"""
for filename in os.listdir(pathDict):
    dictionaries[filename.replace(".pkl","")] = joblib.load(pathDict+"/"+filename)

print("End to load Neural Networks and Dictionaries")


@app.route("/",methods=['GET','POST'])
def classifier():
    if request.method == 'POST':
        data = request.data
        dataDict = json.loads(data)
        text = dataDict.get('text')
    elif request.method == 'GET':
        text = request.args.get('text')
    
	
    """CREO IL BANCH"""
    textToSend = []
    textToSend.append(text)

    dataToTest = Bunch(data=textToSend,filenames="",target="")

    categoriesSimple = Classifier.start(neuralNetworks,dictionaries,dataToTest,10,"root",path)
    json_simple = json.dumps(categoriesSimple)
    body = "{\"simple\":" + json_simple + "}"
    
    response = Response(body,mimetype='application/json')
    response.headers.add('content-length', str(len(body))) 
    response.headers.add('Access-Control-Allow-Origin','*')
 
    return response
    
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(porta))