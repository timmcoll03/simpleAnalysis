import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
import nltk 

data = pd.read_csv("/Users/timothycolledge/Desktop/Compsci/Project 3/simpleAnalysis/Musical_instruments_reviews.csv", index_col=False)
#print(data.head(10))

tokenizer= Tokenizer()

#print(data.reviewText[:10])

tokenizer.fit_on_texts(data.reviewText[:10])

#print(f'ListOfWords: {list(tokenizer.word_index.keys())}')

#print(tokenizer.word_index)

display = tokenizer.texts_to_matrix(data.reviewText[:10], mode='freq')


print(display)

plt.plot()

countData = pd.DataFrame(display)

#print(countData.head(10))

clusters = KMeans(n_clusters=5)
clusters.fit(countData)
#print(clusters.inertia_)

clusteredData = pd.DataFrame(clusters.cluster_centers_)
#print(clusteredData.head())


cluster_map = pd.DataFrame()
cluster_map['data_index'] = countData.index.values
cluster_map['cluster'] = clusters.labels_
#print(cluster_map.head(10))


plt.plot(display)
plt.show()