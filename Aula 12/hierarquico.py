import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

dados = pd.read_csv('../Bases/PessoaNormBinary.csv')

#Hierarchical - Linkage - Complete
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
cluster.fit(dados)

plt.figure(figsize=(10,7))
plt.title("Custumer Dendograms")
dend = shc.dendrogram(shc.linkage(dados, method='complete'))
plt.show()