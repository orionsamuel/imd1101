import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

dados = pd.read_csv('../Bases/PessoaNormBinary.csv')

#Hierarchical - Linkage - Complete
ahc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
ahc.fit(dados)
ahc.fit_predict(dados)

#imprimir os labels
print(ahc.labels_)

#juntando os labels com o restante do dataset
dados["Cluster"] = ahc.labels_

print(dados)