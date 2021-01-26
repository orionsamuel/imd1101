import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

dados = pd.read_csv('../Bases/diabetes.csv')

X = dados.iloc[:,:-1].values
y = dados.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

kf = KFold(n_splits = 10)