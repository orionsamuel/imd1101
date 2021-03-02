import pandas as pd

dados = pd.read_csv("../Bases/Pessoa.csv")

genero = pd.get_dummies(dados['Genero'], prefix='Gen')
estado_civil = pd.get_dummies(dados['Estado_Civil'], prefix='EC')
escolaridade = pd.get_dummies(dados['Escolaridade'], prefix='Esc')
conta_corrente = pd.get_dummies(dados['Conta_Corrente'], prefix='CC')
cartao_credito = pd.get_dummies(dados['Cartao_Credito'], prefix='Cart')
imovel_proprio = pd.get_dummies(dados['Imovel_Proprio'], prefix='Imovel_P')

dados = dados.join(genero)
dados = dados.join(estado_civil)
dados = dados.join(conta_corrente)
dados = dados.join(cartao_credito)
dados = dados.join(imovel_proprio)

dados = dados.drop(["Genero", "Estado_Civil", "Escolaridade", "Conta_Corrente",
"Cartao_Credito", "Imovel_Proprio", "Gen_Femi", "CC_Não", "Imovel_P_Não"], axis = 1)

dados = (dados-dados.min())/(dados.max()-dados.min())

dados.to_csv("PessoaNormBinary.csv", sep=',', index=False)

print(dados)
