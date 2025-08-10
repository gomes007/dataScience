# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 15:43:17 2025

@author: João Mello
"""

import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from funcoes_ajuda import descritiva
# import seaborn as sns

#%% Carregar os dados titanic, já tratada conforme a primeira aula

titanic = pd.read_pickle('titanic1.pkl')

#%% Verificar valores ausentes
print(titanic.isnull().sum())

#%% Gera 80% de treino e 20% de teste
# Separar variáveis preditoras e resposta
X = titanic.drop('survived', axis=1)
y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2360873)

# Sempre importante conferir a cada passo!
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%% Parâmetros do GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [2, 3],
    'gamma': [0],
    'learning_rate': [0.1, 0.4],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [1],
    'subsample': [0.75, 1]
}

#%% Instanciar a classe do XGB
import time
tempo_ini = time.time()

xgb = XGBClassifier(objective='binary:logistic', random_state=2360873)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                           scoring='roc_auc', cv=10, verbose=0, n_jobs=-1)



#%% Treinar o XGB já com o grid search
grid_search.fit(X_train, y_train)

#%% Tempo final
tempo_fim = time.time()
print(f"Tempo de execução: {tempo_fim - tempo_ini} segundos")

#%% Avaliar o XGBoosting
def avalia(modelo, X_train, y_train, X_test, y_test):
    p_train = modelo.predict_proba(X_train)[:, 1]
    # c_train = modelo.predict(X_train)
    
    p_test = modelo.predict_proba(X_test)[:, 1]
    # c_test = modelo.predict(X_test)

    auc_train = roc_auc_score(y_train, p_train)
    auc_test = roc_auc_score(y_test, p_test)
    
    print(f'Avaliação base de treino: AUC = {auc_train:.2f}')
    print(f'Avaliação base de teste: AUC = {auc_test:.2f}')
    
    fpr_train, tpr_train, _ = roc_curve(y_train, p_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, p_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train, color='red', label=f'Treino AUC = {auc_train:.2f}')
    plt.plot(fpr_test, tpr_test, color='blue', label=f'Teste AUC = {auc_test:.2f}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('Falso Positivo')
    plt.ylabel('Verdadeiro Positivo')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

avalia(grid_search.best_estimator_, X_train, y_train, X_test, y_test)

#%% Avaliando a previsão do modelo
titanic['pred'] = grid_search.best_estimator_.predict_proba(X)[:, 1]
descritiva(titanic, 
           'pred',
           vresp='survived', 
           max_classes=10)