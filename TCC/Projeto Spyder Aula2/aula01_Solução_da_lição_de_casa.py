# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:28:03 2025

@author: João Mello
"""

import pandas as pd
from funcoes_ajuda import descritiva, avalia_clf


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet('exercicio.parquet')

print(df.info())
print(df.head())

#%% Descritivas básicas

for var in df.columns:
    descritiva(df, var=var, vresp = 'inadimplencia')
    

#%% Separando treino e teste
y = df['inadimplencia']
X = df.drop('inadimplencia', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2360873)

# Imprima as formas dos conjuntos de dados resultantes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#%% Treinando o modelo

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

#%% Avaliação base de treino
avalia_clf(clf, y_train, X_train, rótulos_y=['Bom', 'Mau'], base = 'treino')

#%% Avaliação base de teste
avalia_clf(clf, y_test, X_test, rótulos_y=['Bom', 'Mau'], base = 'teste')

#%% Obter os valores de CCF desta árvore
ccp_path = pd.DataFrame(clf.cost_complexity_pruning_path(X_train, y_train))

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
plt.show()

print(f'O GINI máximo é de: {GINI_max:.2%}\nObtido com um ccp de: {ccp_max}')

#%% Árvore ótima

arvore_tunada = DecisionTreeClassifier(criterion='gini', max_depth = 30, 
                                random_state=42,
                                ccp_alpha=ccp_max).fit(X_train, y_train)


#%% avaliar

print('Avaliando a base de treino:')
avalia_clf(arvore_tunada, y_train,X_train, base='treino')
print('Avaliando a base de teste:')
avalia_clf(arvore_tunada, y_test,X_test, base='teste')