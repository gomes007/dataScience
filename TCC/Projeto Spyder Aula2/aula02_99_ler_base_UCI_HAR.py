# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 21:38:13 2025

@author: João Mello
"""

#%%
import pandas as pd
import os

#%% Ler os nomes das variáveis

#caminho do arquivo
features_path = os.path.join('UCI HAR Dataset', 'features.txt')

# Ler o arquivo features.txt
features = pd.read_csv(features_path, delim_whitespace=True, header=None, names=['index', 'feature'])

# remover caracteres especiais (isso pode dar uma dor de cabeça depois...)
features = features.feature.str.replace('(','')\
        .str.replace(')','')\
        .str.replace(',','')\
            .to_list()

# Verificando os nomes das variáveis
#%%
nomes_variaveis = [f'{i:03}-{features[i]}' for i in range(len(features))]
#%%

pd.Series(features).duplicated().sum()

#%%
# Definir o caminho dos arquivos de forma portátil
X_train_path = os.path.join('UCI HAR Dataset', 'train', 'X_train.txt')
subject_train_path = os.path.join('UCI HAR Dataset', 'train', 'subject_train.txt')


# Ler o arquivo X_train.txt, especificando o delimitador como espaço
X_train = pd.read_csv(X_train_path, delim_whitespace=True, header=None)

# Ler o arquivo subject_train.txt, especificando o delimitador como espaço
subject_train = pd.read_csv(subject_train_path, delim_whitespace=True, header=None)
subject_train.columns = ['subject']

# Definir os nomes das colunas usando os nomes das variáveis do features.txt
X_train.columns = nomes_variaveis

# juntar X_train com subject_train
X_train = pd.concat([subject_train, X_train], axis=1)#.set_index(['subject'], append=True)

# Exibir as primeiras linhas do dataframe
print(X_train.head(3))


#%% Le y_train
# Definir o caminho dos arquivos de forma portátil
activity_labels_path = os.path.join('UCI HAR Dataset', 'activity_labels.txt')
y_train_path = os.path.join('UCI HAR Dataset', 'train', 'y_train.txt')

# Ler o arquivo features.txt
activity_labels = pd.read_csv(activity_labels_path, 
                              delim_whitespace=True, 
                              header=None, 
                              index_col = 0,
                              names=['label'])

# Ler o arquivo X_train.txt, especificando o delimitador como espaço
y_train = pd.read_csv(y_train_path, delim_whitespace=True, header=None)

# Definir os nomes das colunas usando os nomes das variáveis do features.txt
activity_labels = activity_labels.to_dict()['label']

y_train.columns = ['label']
y_train.label.dtypes
activity_labels

# TRansforma y_train em categorical, com os rótulos corretos
y_train['label'] = pd.Categorical(y_train.label.map(activity_labels), ordered=False)

y_train.value_counts(dropna=False)

#%% Ler as bases de testes
X_test_path = os.path.join('UCI HAR Dataset', 'test', 'X_test.txt')
subject_test_path = os.path.join('UCI HAR Dataset', 'test', 'subject_test.txt')


# Ler o arquivo X_test.txt, especificando o delimitador como espaço
X_test = pd.read_csv(X_test_path, delim_whitespace=True, header=None)

# Ler o arquivo subject_test.txt, especificando o delimitador como espaço
subject_test = pd.read_csv(subject_test_path, delim_whitespace=True, header=None)
subject_test.columns = ['subject']

# Definir os nomes das colunas usando os nomes das variáveis do features.txt
X_test.columns = nomes_variaveis

# juntar X_train com subject_train
X_test = pd.concat([subject_test, X_test], axis=1)#.set_index(['subject'], append=True)

# Exibir as primeiras linhas do dataframe
print(X_test.head())

#%% Le y_train
# Definir o caminho dos arquivos de forma portátil
y_test_path = os.path.join('UCI HAR Dataset', 'test', 'y_test.txt')

# Ler o arquivo X_test.txt, especificando o delimitador como espaço
y_test = pd.read_csv(y_test_path, delim_whitespace=True, header=None)

y_test.columns = ['label']

# TRansforma y_train em categorical, com os rótulos corretos
y_test['label'] = pd.Categorical(y_test.label.map(activity_labels), ordered=False)

y_test.value_counts(dropna=False)

#%% Padronizar nomes das variáveis
X_test.columns


#%% Salvar em pickle

X_train.to_pickle('X_train.pkl')
y_train.to_pickle('y_train.pkl')
X_test.to_pickle('X_test.pkl')
y_test.to_pickle('y_test.pkl')