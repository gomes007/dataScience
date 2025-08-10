# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:32:51 2024

@author: João Mello
"""

#%% Instalar pacote adicional

!pip install patsy

#%% Carregar bibliotecas
import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from funcoes_ajuda import avalia_clf


import patsy
import time
#%% Ler a base de dados
titanic = sns.load_dataset('titanic')

titanic.head()
#%% Criar a matriz de dados X e a variável resposta y usando patsy
y, X = patsy.dmatrices('survived ~ pclass + sex + age + sibsp + parch + fare + embarked', data=titanic, return_type="dataframe")

# Exiba as primeiras linhas da matriz de dados X
print(X.head())

# Exiba as primeiras linhas da variável resposta y
print(y.head())
#%% Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2360873)

# Imprima as formas dos conjuntos de dados resultantes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
#%% Treinar a Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

#%% # Criar data-frame de avaliação de treino e teste

avalia_clf(rf, y_train, X_train, rótulos_y=['Não Sobreviveu', 'Sobreviveu'], base = 'treino')
avalia_clf(rf, y_test, X_test, rótulos_y=['Não Sobreviveu', 'Sobreviveu'], base = 'teste')

#%% Montando a estrutura para um gridsearch

tempo_ini = time.time()

param_grid = {'n_estimators': [100], 'max_features': range(1, 11)}

rf_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, 
                           param_grid=param_grid, 
                           scoring='roc_auc', 
                           cv=4, 
                           n_jobs=-1)

grid_search.fit(X_train, y_train.values.ravel()) 

# Print the best parameters and the best score
print(grid_search)
print(grid_search.best_params_)
print(grid_search.best_score_)
tempo_fim = time.time()

melhor_modelo = grid_search.best_estimator_

print(f"Tempo de execução: {tempo_fim - tempo_ini} segundos")

#%% Avaliar o modelo tunado
# aval_classificador(y_train, X_train, y_test, X_test, melhor_modelo)


avalia_clf(melhor_modelo, y_train, X_train, rótulos_y=['Não Sobreviveu', 'Sobreviveu'], base = 'treino')
avalia_clf(melhor_modelo, y_test, X_test, rótulos_y=['Não Sobreviveu', 'Sobreviveu'], base = 'teste')