# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:26:07 2025

@author: João Mello
"""
#%% Ler bibliotecas
import pandas as pd


#%% LIÇÃO DE CASA

# Lição de casa - Tente ajustar um modelo, com o que você aprendeu, nas bases UCI HAR

X_train = pd.read_pickle('X_train.pkl')
X_test = pd.read_pickle('X_test.pkl')
y_train = pd.read_pickle('y_train.pkl')
y_test = pd.read_pickle('y_test.pkl')


#%%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)