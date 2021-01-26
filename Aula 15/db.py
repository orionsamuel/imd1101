import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

dados = pd.read_csv('../Bases/PessoaNormBinary.csv')

km = KMeans(n_clusters=3, random_state=1)
km.fit(dados)
labels = km.labels_

print(davies_bouldin_score(dados, labels))