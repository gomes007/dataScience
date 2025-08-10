# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:22:11 2025

@author: João Mello
"""
!pip install shap --upgrade
#%%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from funcoes_ajuda import descritiva, diagnóstico

import shap
import matplotlib.pyplot as plt


df = pd.read_csv('houses_to_rent.csv', index_col=0)


df.info()

df['property tax'] = df['property tax']

#%%
df['floor'] = df.floor.str.replace('-','NaN').astype('float64')
for var in ['hoa', 'rent amount', 'property tax', 'fire insurance', 'total']:
    df[var] = df[var].str.replace('R$','')\
        .str.replace(',','')\
        .str.replace('Sem info','NaN')\
        .str.replace('Incluso','0').astype('float64')
        
df.info()

X_cols = ['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'animal', 'furniture']
y_col = 'total'

X = pd.get_dummies(df[X_cols], drop_first=True)
y = df[y_col]

for col in X_cols:
    descritiva(df, col, y)


#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%%
rf = RandomForestRegressor()
rf.fit(X, y)

#%%
r2_score(y_test, rf.predict(X_test))

#%%
df['pred'] = rf.predict(X)

#%% Vamos rodar uma rotina de diagnóstico por variáveis

# Esta análise vai mostrar valores esperados vs observados por cada variável
for col in X_cols:
    diagnóstico(df, col, y, 'pred')
    
#%% Calcular os 'shap values'
amostra = X_test.sample(frac=0.1)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(amostra)

#%% Gráfico Resumo
shap.summary_plot(shap_values, amostra, feature_names=X.columns)

#%% Gráfico de cascata no nível indivíduo
shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                     base_values=explainer.expected_value, 
                                     data=amostra.iloc[0], 
                                     feature_names=amostra.columns))


#%% Fazendo o waterfall 'na mão'
df_shap = pd.DataFrame(shap_values, columns = X.columns)
df_shap.iloc[0].plot.bar()


#%% Forceplot
# Inicializar a visualização
shap.initjs()

# Explicação no nível de indivíduo (force plot para a primeira amostra de teste)
force_plot = shap.force_plot(explainer.expected_value, 
                shap_values[0], 
                amostra.iloc[0], 
                feature_names=amostra.columns)
plt.show()

# Este não mostra no console, vamos salvar em arquivo
shap.save_html("force_plot.html", force_plot)

#%%
