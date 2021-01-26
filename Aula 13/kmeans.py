import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dados = pd.read_csv('../Bases/PessoaNormBinary.csv')

#k-Means
km = KMeans(n_clusters=3)
km.fit(dados)
centroids = km.cluster_centers_

plt.scatter(dados.iloc[:,0], dados.iloc[:,1])
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=300)
plt.show()