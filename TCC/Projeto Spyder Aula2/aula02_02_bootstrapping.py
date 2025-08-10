# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:32:51 2024

@author: João Mello
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Gera os dados
# Gerar um dataframe com dados aleatórios
np.random.seed(123)  # Definir a semente para reprodução dos resultados
# Exemplo: dataframe com uma coluna chamada "dados" e 100 observações
df = pd.DataFrame({'dados': np.random.normal(size=100)})

#%% Calcula o erro padrão por bootstrap
# Função para calcular o erro padrão da média
def calcular_erro_padrao_media(data, n_boot=1000):
  n = len(data)
  medias_boot = np.zeros(n_boot)  # Vetor para armazenar as médias bootstrap
  for i in range(n_boot):
    # Amostragem bootstrap - com reposição - mesmo tamanho da amostra
    bootstrap_sample = np.random.choice(data, size=n, replace=True)
    # Cálculo da média da amostra bootstrap
    medias_boot[i] = np.mean(bootstrap_sample)
  return medias_boot

#%% Chamada da função para calcular o erro padrão da média no dataframe
amostra_bootstrap = calcular_erro_padrao_media(df['dados'])

#%%  Histograma das médias bootstrap
plt.hist(amostra_bootstrap)
plt.title('Histograma das médias bootstrap')
plt.xlabel('Médias Bootstrap')
plt.ylabel('Frequência')
plt.show()

# Desvio padrão das médias bootstrap
desvio_padrao = np.std(amostra_bootstrap)
print(f"Desvio padrão das médias bootstrap: {desvio_padrao}")