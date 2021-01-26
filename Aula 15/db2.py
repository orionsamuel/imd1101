import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score

dados = pd.read_csv('../Bases/PessoaNormBinary.csv')

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
cluster.fit(dados)

unique, counts = np.unique(cluster.labels_, return_counts=True)
dict(zip(unique, counts))

#DB
plt.figure(figsize=(10,7))
plt.title("Davies Bouldin Coefficient (k = 3, seed = 10): %0.3f" 
% davies_bouldin_score(dados, cluster.labels_))
plt.scatter(dados.iloc[:,0], dados.iloc[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()

cluster.fit_predict(dados)

#Jutando as labels com o restante do dataset
dados["Cluster"] = cluster.labels_
dados["Cluster"] = 'cluster' + dados["Cluster"].astype(str)

#Salvando a base transformada
dados.to_csv("PessoaNormBinary_Ha-clink-3k.csv", sep=',', index=False)