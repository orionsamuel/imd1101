import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture #EM
from sklearn import metrics

dados = pd.read_csv('../Bases/PessoaNormBinary.csv')

gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(dados)

dados["Cluster"] = gmm.predict(dados)
dados["Cluster"] = 'cluster' + dados["Cluster"].astype(str)

print(dados)