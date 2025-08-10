# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:26:37 2025

@author: João Mello
"""


#%% 
# Importar os pacotes necessários
import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor, plot_tree
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV

#%% 
# Carregar o Ames Housing Dataset
url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
ames_housing = pd.read_csv(url)

# Visualizar as primeiras linhas da base de dados
print(ames_housing.head())

###################################################################
### Lição de casa
### 
### Tente construir um modelo preditivo para o valor do imóvel
### target: median_house_value
###################################################################
