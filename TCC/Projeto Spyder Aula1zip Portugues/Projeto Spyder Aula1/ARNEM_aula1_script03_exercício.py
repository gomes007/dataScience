# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 16:56:06 2025

@author: João Mello
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import random

# Definir uma semente aleatória para reprodutibilidade
np.random.seed(42)
random.seed(42)

# Gerar as variáveis simuladas com correlação
idade = np.random.randint(18, 71, 10000)

# Gerar variáveis correlacionadas usando a função multivariada normal
mean_values = [5000, 2000, 0.5, 5]  # Médias das variáveis
correlation_matrix = np.array([
    [1, 0.3, 0.2, -0.1],
    [0.3, 1, -0.1, 0.2],
    [0.2, -0.1, 1, 0.4],
    [-0.1, 0.2, 0.4, 1]
])  # Matriz de correlação

# Gerar dados simulados
simulated_data = np.random.multivariate_normal(mean_values, correlation_matrix, 10000)

renda = simulated_data[:, 0]
divida = simulated_data[:, 1]
utilizacao_credito = np.clip(simulated_data[:, 2], 0, 1)  # Limita a utilização de crédito entre 0 e 1
consultas_recentes = np.maximum(simulated_data[:, 3], 0)  # Garante que o número de consultas recentes seja não negativo

# Gerar função linear das variáveis explicativas
preditor_linear = -7 - 0.01 * idade - 0.0002 * renda + 0.003 * divida - 3 * utilizacao_credito + 0.5 * consultas_recentes

# Calcular probabilidade de default (PD) usando a função de link logit
prob_default = 1 / (1 + np.exp(-preditor_linear))

# Gerar inadimplência como variável Bernoulli com base na probabilidade de default
inadimplencia = np.random.binomial(1, prob_default, 10000)

# Criar dataframe
dados = pd.DataFrame({
    'idade': idade,
    'renda': renda,
    'divida': divida,
    'utilizacao_credito': utilizacao_credito,
    'consultas_recentes': consultas_recentes,
    'inadimplencia': inadimplencia
})

print(dados.head())

# Categorizar a idade
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
dados['idade_cat'] = kbin.fit_transform(dados[['idade']])

def descritiva2(var1, var2, df):
    cross_tab = pd.crosstab(df[var1], df[var2], normalize='index')
    print(cross_tab)

descritiva2('idade_cat', 'inadimplencia', dados)

print(dados.head())

dados.to_parquet('exercicio.parquet')

#####################################################################################
# Agora é a sua vez: Ajuste uma árvore de decisão, e explore os recursos que fizemos 
# ao longo da aula nesta base de dados ;)
