# -*- coding: utf-8 -*-
"""
Editor Spyder

Árvores
Ensemble
Machine Learning
Redes Neurais
"""

# Importações necessárias no arquivo 00
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Função para separar treino e teste
# Métricas de avaliação do modelo programadas no scikit
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score

# Classe de árvore e funções auxiliares
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

#%%  Funções Auxiliares

from funcoes_ajuda import descritiva, relatorio_missing

#%%  Carregar os dados
titanic = sns.load_dataset('titanic')

print(titanic.head())
print(titanic.columns)

#%%  Análise descritiva básica

for variavel in titanic.columns:
    print(f'\n\nAnálise univariada de {variavel}:')
    print(titanic[variavel].describe())

#%%

for variavel in ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'class', 'who', 
                 'adult_male', 'deck', 'embark_town', 'alive', 'alone']:
    print(f'\n\nFrequencias da variável: {variavel}')
    print(titanic[variavel].value_counts(dropna=False).sort_index())
    
#%%
descritiva(titanic, "sex")
descritiva(titanic, "class")
descritiva(titanic, "age", max_classes=10)
descritiva(titanic, "fare", max_classes=5)
descritiva(titanic,"embarked")
descritiva(titanic,"sibsp")
descritiva(titanic,"parch")

#%% Avaliar dados faltantes

# A função tem basicamenteum compilado desses comandos:
# titanic.isna().sum()
# titanic.isna().mean().apply(lambda x: f"{x:.1%}".)
relatorio_missing(titanic)

#%% Tratar variável age

titanic['age'] = titanic.age.fillna(titanic.age.mean())

#%% Remover variáveis redundantes
titanic.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town', 
                      'alive', 'alone'], inplace=True)
#%% Verificar variáveis string

metadados = titanic.dtypes

print('\nVariávels string:')
print(metadados[metadados == 'object'])

metadados

#%% Transformar variáveis string em dummies

# No pandas, um método prático de fazer isto é com o get_dummies
titanic_dummies = pd.get_dummies(titanic, drop_first=True)

# Conferir a estrutura da tabela
titanic_dummies.info()
# Checar primeiras 3 linhas
titanic_dummies.head(3)
    
#%% Salvando a base tratada

# O pkl é um formato do Python, que vai manter todas as propriedades do objeto
titanic_dummies.to_pickle('titanic1.pkl')
# Parquet é um formato popular, eficiente, compatível com outras plataformas
titanic_dummies.to_parquet('titanic1.parquet')

#%%  Rodando a primeira árvore

# No Scikitlearn, vamos usar a seguinte estrutura:
    
# Um objeto X com as variáveis explicativas (miúsculo por serem várias)
X = titanic_dummies.drop(columns = ['survived'])
# Um objeto y com a variável resposta (target) minúsculo por ser só 1
y = titanic_dummies['survived']

# Estanciar a classe da árvore de decisão em um objeto chamado arvore
# Este objeto não possui os dados em um primeiro momento
# Mas ela possui todos os atributos e métodos que precisaremos
arvore = DecisionTreeClassifier(criterion='gini', max_depth = 3, random_state=42)

# Treinar o modelo com o método .fit(). Aqui processamos os dados!
arvore.fit(X, y)

# OBS: O objeto árvore contémodos os atributos e métodos que precisamos
# Vamos usar esse objeto para várias coisas como:
#    visualizar as regras da árvore, avaliar a árvore, classificar novas linhas

#%%  Plotar a árvore
plt.figure(figsize=(20, 10))
plot_tree(arvore, feature_names=X.columns.tolist(), class_names=['Not Survived', 'Survived'], filled=True)
plt.show()

#%%  Novos dados

# Suponha que temos novos dados para classiicar
novos_dados = X.tail() # como exemplo, vamos classificar as 5 últimas linhas
print(novos_dados)

#%%  Classificando com a árvore

# Predict é o método que classifica as novas observações
#    Lembrete: a classificação é pela classe mais frequente da folha
classificação_novos_dados = arvore.predict(novos_dados)
classificação_novos_dados

#%%  Avaliando a classificação
# Vamos comparar a classificação da árvore com o valor observado

# Guardar a classificação da árvore 
classificação_treino = arvore.predict(X)

# Comparar com os valores reais por uma tabela cruzada
print(pd.crosstab(classificação_treino, y, margins=True))
print(pd.crosstab(classificação_treino, y, normalize='index'))
print(pd.crosstab(classificação_treino, y, normalize='columns'))

acertos = classificação_treino == y
pct_acertos = acertos.sum()/acertos.shape[0]
print(f'Acurácia (taxa de acerto): {pct_acertos:.2%}')

#%% 

# Calculando acurácia e matriz de confusão

# Vamos avaliar o modelo com algumas funções próprias do Scikit-Learn
# A função confudion_matrix faz basicamente a comparação acima
cm = confusion_matrix(y, arvore.predict(X))
# accuracy_score calcula o percentual de acertos
ac = accuracy_score(y, arvore.predict(X))
# Essa função pondera para forçar a distribuição da target como uniforme
bac = balanced_accuracy_score(y, arvore.predict(X))

print(f'\nA acurácia da árvore é: {ac:.1%}')
print(f'A acurácia balanceada da árvore é: {bac:.1%}')

# Visualização gráfica
sns.heatmap(cm, 
            annot=True, fmt='d', cmap='viridis', 
            xticklabels=['Não Sobreviveu', 'Sobreviveu'], 
            yticklabels=['Não Sobreviveu', 'Sobreviveu'])
plt.show()

# Relatório de classificação do Scikit
print('\n', classification_report(y, arvore.predict(X)))



