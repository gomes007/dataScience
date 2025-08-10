# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:36:35 2025

@author: João Mello
"""
import seaborn as sns
import matplotlib.pyplot as plt

# Alguns pacotes que você talvez possa precisar:
    
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import GridSearchCV


#%% Carregar os dados (assumindo que 'tips.csv' tem a mesma estrutura)
tips = sns.load_dataset('tips')

#%% Criar a coluna de percentual de gorjeta
tips['pct_tip'] = tips['tip'] / (tips['total_bill'] - tips['tip'])

#%% Exploração dos dados
sns.boxplot(x='pct_tip', data=tips)
plt.show()

#%% Remover outliers (opcional)
tips = tips[tips['pct_tip'] < 1]

#%% ##############################
### Lição de casa: Ajuste um modelo buscando prever o valor de pct_tip