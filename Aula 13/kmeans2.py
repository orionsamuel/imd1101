import pandas as pd
from sklearn.cluster import KMeans

dados = pd.read_csv('../Bases/PessoaNormBinary.csv')

#k-Means
km = KMeans(n_clusters=3, init='k-means++', max_iter = 300, n_init=10, random_state = 0)
km.fit(dados)
km.fit_predict(dados)

dados["Cluster"] = km.labels_
dados["Cluster"] = 'cluster' + dados["Cluster"].astype(str)

dados.to_csv("Pessoa_Clustered_kM_k3.csv", sep=',', index=False)

print(dados)