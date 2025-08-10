# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:36:35 2025

@author: João Mello
"""

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from funcoes_ajuda import descritiva

#%% Carregar os dados (assumindo que 'tips.csv' tem a mesma estrutura)
tips = sns.load_dataset('tips')

#%% Criar a coluna de percentual de gorjeta
tips['pct_tip'] = tips['tip'] / (tips['total_bill'] - tips['tip'])

#%% Exploração dos dados
sns.boxplot(x='pct_tip', data=tips)
plt.show()

#%% Remover outliers (opcional)
tips = tips[tips['pct_tip'] < 1]

#%% Preparar os dados para o modelo
X = tips[['sex', 'smoker', 'day', 'time', 'size', 'total_bill']]
y = tips['pct_tip']

#%% Análises descritivas

for col in tips.columns[:-1]:
    descritiva(df_=tips, var=col, vresp='pct_tip')

#%% Codificar variáveis categóricas (se necessário)
X = pd.get_dummies(X)

#%% Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Criar o modelo de árvore de decisão
model = RandomForestRegressor(random_state=42)

#%% Treinar o modelo
model.fit(X_train, y_train)

#%% Fazer previsões
y_pred = model.predict(X_test)

#%% Avaliar o modelo (usando MSE como exemplo)
mse = mean_squared_error(y_test, y_pred)
rquad = r2_score(y_test,y_pred)
print(f'Mean Squared Error: {mse:.5f} | R-quadrado={rquad:,.1f}')

#%% Otimização do modelo (usando cross-validation)
# Definir os parâmetros para o grid search 

params = { 'n_estimators': [100], 
          'max_depth': [2, 3, 6], 
          'min_samples_split': [2, 5],
          'max_features': [2, 5]
          }

grid = GridSearchCV(RandomForestRegressor(), 
                    params, 
                    cv=5)

grid.fit(X_train, y_train)
print(grid.best_params_)

#%% Avaliando o modelo na base de teste
y_pred = grid.best_estimator_.predict(X_test)

r2treino = r2_score(y_train,y_train)
r2teste = r2_score(y_test,y_pred)

print(f"R-quadrado na base de teste: {r2teste:,.2%}")

#%% Visualização gráfica do resultado
sns.pointplot(x=pd.qcut(y_pred, 5, duplicates='drop'), y=y_test)
plt.show()

#Vamos pensar: Qual o tamanho da base? Quantas observações ficam para cada 'fold'?
