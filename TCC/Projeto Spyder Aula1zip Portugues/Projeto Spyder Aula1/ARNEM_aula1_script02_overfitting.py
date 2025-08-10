# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:08:12 2024

@author: João Mello
"""

import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from funcoes_ajuda import avalia_clf

#%% Carregando a base

titanic = pd.read_pickle('titanic1.pkl')
# titanic = pd.read_parquet('titanic1.parquet')

#%%  Dividindo a base em treino e teste

# Selecionar variáveis preditoras e a variável resposta
X = titanic.drop(columns = ['survived'])
y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

# conferindo número de linhas e colunas
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%% Treinando a árvore

# Criar o modelo de árvore de decisão
arvore = DecisionTreeClassifier(criterion='gini', 
                                max_depth = 3, 
                                random_state=42)

# Treinar o modelo
arvore.fit(X_train, y_train)

#%% Avaliando o modelo na base de treino

print('Avaliando a base de treino:')
avalia_clf(arvore, y_train,X_train)


#%% Avaliando o modelo na base de teste
print('Avaliando a base de teste:')
avalia_clf(arvore, y_test,X_test)

#%%  Deixando a árvore ser feliz

arvore = DecisionTreeClassifier(criterion='gini', max_depth = 30, 
                                random_state=42,
                                ccp_alpha=0)

# Treinar o modelo
arvore.fit(X_train, y_train)

#%% Avaliando o modelo na base de treino e teste

print('Avaliando a base de treino:')
avalia_clf(arvore, y_train,X_train, base='treino')
print('Avaliando a base de teste:')
avalia_clf(arvore, y_test,X_test, base='teste')


#%% ccp_alpha

ccp_path = pd.DataFrame(arvore.cost_complexity_pruning_path(X_train, y_train))


#%% Tunando a árvore

GINIs = []

for ccp in ccp_path['ccp_alphas']:
    arvore = DecisionTreeClassifier(criterion='gini', max_depth = 30, 
                                    random_state=42,
                                    ccp_alpha=ccp)

    # Treinar o modelo
    arvore.fit(X_train, y_train)
    AUC = roc_auc_score(y_test, arvore.predict_proba(X_test)[:, -1])
    GINI = (AUC-0.5)*2
    GINIs.append(GINI)

sns.lineplot(x = ccp_path['ccp_alphas'], y = GINIs)

df_avaliacoes = pd.DataFrame({'ccp': ccp_path['ccp_alphas'], 'GINI': GINIs})

GINI_max = df_avaliacoes.GINI.max()
ccp_max  = df_avaliacoes.loc[df_avaliacoes.GINI == GINI_max, 'ccp'].values[0]

plt.ylabel('GINI da árvore')
plt.xlabel('CCP Alphas')
plt.title('Avaliação da árvore por valor de CCP-Alpha')

print(f'O GINI máximo é de: {GINI_max:.2%}\nObtido com um ccp de: {ccp_max}')

#%% Árvore ótima

arvore = DecisionTreeClassifier(criterion='gini', max_depth = 30, 
                                random_state=42,
                                ccp_alpha=ccp_max).fit(X_train, y_train)


#%% avaliar

print('Avaliando a base de treino:')
avalia_clf(arvore, y_train,X_train, base='treino')
print('Avaliando a base de teste:')
avalia_clf(arvore, y_test,X_test, base='teste')