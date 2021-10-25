# Timothy Colledge, Jeffery Cheng, and Kevin Cavicchia 
# Find ReadME for more documentation

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
import nltk 
import csv


def kmeans(dataSet):#Clustering analysis
    
    #Fitting Data and Clustering
    countData = pd.DataFrame(dataSet)
    clusters = KMeans(n_clusters=5)
    clusters.fit(countData)
    
    #Creating map of cluster centers and data 
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = countData.index.values
    cluster_map['cluster'] = clusters.labels_
    
    #Outputting readable information on model
    print(clusters.inertia_)
    print(cluster_map.head())
    plt.plot(cluster_map.data_index,cluster_map.cluster)
    plt.show()


def keras():#Keras Tokenization and Freq. Analysis
    
    #Pre-processing of data and Tokenizing
    data = pd.read_csv("/Users/timothycolledge/Desktop/Compsci/Project 3/simpleAnalysis/Musical_instruments_reviews.csv", index_col=False)
    tokenizer= Tokenizer()
    tokenizer.fit_on_texts(data.reviewText[:300])

    #Freq. Analysis
    display = tokenizer.texts_to_matrix(data.reviewText[:300], mode='freq')

    #Output Checks
    print(data.head(10))
    print(data.reviewText[:10])
    print(f'ListOfWords: {list(tokenizer.word_index.keys())}')
    print(tokenizer.word_index)
    plt.plot(display[0])
    plt.show()

    kmeans(display)

def nltk():#nltk Tokenization and Freq. Analysis

    #Pre-processing of data and Tokenizing
    nldata= []
    with open("/Users/timothycolledge/Desktop/Compsci/Project 3/simpleAnalysis/Musical_instruments_reviews.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            nldata.append(nltk.word_tokenize(row["reviewText"]))

    #Freq. Analysis
    nlCountData = []
    nlKmeansData = {}
    for x in range (len(nldata)):  
        nlcountData = {}
        for nam in nldata[x]:  
            if nam not in nlcountData:
                nlcountData.update({nam: 0})
                nlKmeansData.update({x: 0})
        nlCountData.append(nlcountData)

    for x in range (len(nldata)):  
        for nam in nldata[x]: 
            nlCountData[x][nam] += 1

    for x in range (len(nldata)):  
        for nam in nldata[x]: 
            nlKmeansData[x] = nlCountData[x][nam]

    nlfinalData = sorted(nlKmeansData.items())
    
    #Output Checks
    x,y = zip(*nlfinalData)
    plt.plot(x,y)
    plt.show()
    
    kmeans(nlfinalData)



analysisType = input("Would you like to do nltk[0], keras[1], or both[2] ")#User Chooses Analysis Type

if(analysisType==str(0)):
    nltk()
if(analysisType==str(1)):
    keras()
if(analysisType==str(2)):
    nltk()
    keras()