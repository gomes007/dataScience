# -*- coding: utf-8 -*-

# MBA Data Science e Analytics USP/ESALQ 
# Support Vector Machine (SVM)

# Prof. Dr. Wilson Tarantin Jr.

#%% Instalar os pacotes necessários

!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install scipy

#%% Importando os pacotes

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

#%% Carregando o banco de dados

dados = pd.read_excel('dados_churn.xlsx')

# Target: churn -> 1: deixou de ser cliente; 0 continua cliente

#%% Estatísticas descritivas

# Variáveis métricas
print(dados[['idade', 'acessos_mes', 'valor_medio_transacao']].describe())

# Variáveis categóricas
print(dados['churn'].value_counts())
print(dados['sexo'].value_counts())
print(dados['cidade'].value_counts())
print(dados['categoria_plano'].value_counts())
print(dados['reclamacao'].value_counts())

#%% Criação das variáveis binárias

dados = pd.get_dummies(dados, 
                       columns=['sexo',
                                'cidade',
                                'categoria_plano',
                                'reclamacao'],
                        drop_first=False,
                        dtype='float')

#%% Separação das bases de treino e teste

# Separando as variáveis Y e X
X = dados.drop(columns=['churn', 'sexo_feminino', 'cidade_interior', 'reclamacao_nao'])
y = dados['churn']

# Gerando os conjuntos de treinamento e testes (70% treino e 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=100)

#%% Padronização das variáveis métricas

# Vamos armazenar as informações de média e desvio padrão para previsões
media_idade, dp_idade = (X_train['idade'].mean(), X_train['idade'].std(ddof=1))
media_acessos, dp_acessos = (X_train['acessos_mes'].mean(), X_train['acessos_mes'].std(ddof=1))
media_valor, dp_valor = (X_train['valor_medio_transacao'].mean(), X_train['valor_medio_transacao'].std(ddof=1))

# No banco de dados de treino
X_train['idade'] = stats.zscore(X_train['idade'], ddof=1)
X_train['acessos_mes'] = stats.zscore(X_train['acessos_mes'], ddof=1)
X_train['valor_medio_transacao'] = stats.zscore(X_train['valor_medio_transacao'], ddof=1)

# No banco de dados de teste
X_test['idade'] = stats.zscore(X_test['idade'], ddof=1)
X_test['acessos_mes'] = stats.zscore(X_test['acessos_mes'], ddof=1)
X_test['valor_medio_transacao'] = stats.zscore(X_test['valor_medio_transacao'], ddof=1)

#%%############### Estimando o modelo SVM (Linear) ############################
###############################################################################

svm_linear = SVC(kernel='linear', C=0.1)
svm_linear.fit(X_train, y_train)

#%% Categorias preditas pelo modelo linear

# Base de dados de treinamento
pred_linear_train = svm_linear.predict(X_train)

# Base de dados de teste
pred_linear_test = svm_linear.predict(X_test)

#%% Análise pela matriz de confusão (base de treino)

cm_linear_train = confusion_matrix(pred_linear_train, y_train)
cm_train_disp_linear = ConfusionMatrixDisplay(cm_linear_train)

plt.rcParams['figure.dpi'] = 600
cm_train_disp_linear.plot(colorbar=False, cmap='Blues')
plt.title('SVM Linear: Treino')
plt.xlabel('Observado (Real)')
plt.ylabel('Classificado (Modelo)')
plt.show()

acc_linear_train = accuracy_score(y_train, pred_linear_train)
sens_linear_train = recall_score(y_train, pred_linear_train, pos_label=1)
espec_linear_train = recall_score(y_train, pred_linear_train, pos_label=0)
prec_linear_train = precision_score(y_train, pred_linear_train)

print("Avaliação do SVM Linear (Base de Treino)")
print(f"Acurácia: {acc_linear_train:.1%}")
print(f"Sensibilidade: {sens_linear_train:.1%}")
print(f"Especificidade: {espec_linear_train:.1%}")
print(f"Precision: {prec_linear_train:.1%}")

#%% Análise pela matriz de confusão (base de teste)

cm_linear_test = confusion_matrix(pred_linear_test, y_test)
cm_test_disp_linear = ConfusionMatrixDisplay(cm_linear_test)

plt.rcParams['figure.dpi'] = 600
cm_test_disp_linear.plot(colorbar=False, cmap='Blues')
plt.title('SVM Linear: Teste')
plt.xlabel('Observado (Real)')
plt.ylabel('Classificado (Modelo)')
plt.show()

acc_linear_test = accuracy_score(y_test, pred_linear_test)
sens_linear_test = recall_score(y_test, pred_linear_test, pos_label=1)
espec_linear_test = recall_score(y_test, pred_linear_test, pos_label=0)
prec_linear_test = precision_score(y_test, pred_linear_test)

print("Avaliação do SVM Linear (Base de Teste)")
print(f"Acurácia: {acc_linear_test:.1%}")
print(f"Sensibilidade: {sens_linear_test:.1%}")
print(f"Especificidade: {espec_linear_test:.1%}")
print(f"Precision: {prec_linear_test:.1%}")

#%%############### Estimando o modelo SVM (Polinomial) ########################
###############################################################################

svm_pol = SVC(kernel='poly', degree=2, coef0=0, C=0.1)
svm_pol.fit(X_train, y_train)

#%% Categorias preditas pelo modelo Polinomial

# Base de dados de treinamento
pred_pol_train = svm_pol.predict(X_train)

# Base de dados de teste
pred_pol_test = svm_pol.predict(X_test)

#%% Análise pela matriz de confusão (base de treino)

cm_pol_train = confusion_matrix(pred_pol_train, y_train)
cm_train_disp_pol = ConfusionMatrixDisplay(cm_pol_train)

plt.rcParams['figure.dpi'] = 600
cm_train_disp_pol.plot(colorbar=False, cmap='gray')
plt.title('SVM Polinomial: Treino')
plt.xlabel('Observado (Real)')
plt.ylabel('Classificado (Modelo)')
plt.show()

acc_pol_train = accuracy_score(y_train, pred_pol_train)
sens_pol_train = recall_score(y_train, pred_pol_train, pos_label=1)
espec_pol_train = recall_score(y_train, pred_pol_train, pos_label=0)
prec_pol_train = precision_score(y_train, pred_pol_train)

print("Avaliação do SVM Polinomial (Base de Treino)")
print(f"Acurácia: {acc_pol_train:.1%}")
print(f"Sensibilidade: {sens_pol_train:.1%}")
print(f"Especificidade: {espec_pol_train:.1%}")
print(f"Precision: {prec_pol_train:.1%}")

#%% Análise pela matriz de confusão (base de teste)

cm_pol_test = confusion_matrix(pred_pol_test, y_test)
cm_test_disp_pol = ConfusionMatrixDisplay(cm_pol_test)

plt.rcParams['figure.dpi'] = 600
cm_test_disp_pol.plot(colorbar=False, cmap='gray')
plt.title('SVM Polinomial: Teste')
plt.xlabel('Observado (Real)')
plt.ylabel('Classificado (Modelo)')
plt.show()

acc_pol_test = accuracy_score(y_test, pred_pol_test)
sens_pol_test = recall_score(y_test, pred_pol_test, pos_label=1)
espec_pol_test = recall_score(y_test, pred_pol_test, pos_label=0)
prec_pol_test = precision_score(y_test, pred_pol_test)

print("Avaliação do SVM Polinomial (Base de Teste)")
print(f"Acurácia: {acc_pol_test:.1%}")
print(f"Sensibilidade: {sens_pol_test:.1%}")
print(f"Especificidade: {espec_pol_test:.1%}")
print(f"Precision: {prec_pol_test:.1%}")

#%%############### Estimando o modelo SVM (RBF) ###############################
###############################################################################

svm_rbf = SVC(kernel='rbf', C=0.1, gamma=1)
svm_rbf.fit(X_train, y_train)

#%% Categorias preditas pelo modelo RBF

# Base de dados de treinamento
pred_rbf_train = svm_rbf.predict(X_train)

# Base de dados de teste
pred_rbf_test = svm_rbf.predict(X_test)

#%% Análise pela matriz de confusão (base de treino)

cm_rbf_train = confusion_matrix(pred_rbf_train, y_train)
cm_train_disp_rbf = ConfusionMatrixDisplay(cm_rbf_train)

plt.rcParams['figure.dpi'] = 600
cm_train_disp_rbf.plot(colorbar=False, cmap='Oranges')
plt.title('SVM RBF: Treino')
plt.xlabel('Observado (Real)')
plt.ylabel('Classificado (Modelo)')
plt.show()

acc_rbf_train = accuracy_score(y_train, pred_rbf_train)
sens_rbf_train = recall_score(y_train, pred_rbf_train, pos_label=1)
espec_rbf_train = recall_score(y_train, pred_rbf_train, pos_label=0)
prec_rbf_train = precision_score(y_train, pred_rbf_train)

print("Avaliação do SVM RBF (Base de Treino)")
print(f"Acurácia: {acc_rbf_train:.1%}")
print(f"Sensibilidade: {sens_rbf_train:.1%}")
print(f"Especificidade: {espec_rbf_train:.1%}")
print(f"Precision: {prec_rbf_train:.1%}")

#%% Análise pela matriz de confusão (base de teste)

cm_rbf_test = confusion_matrix(pred_rbf_test, y_test)
cm_test_disp_rbf = ConfusionMatrixDisplay(cm_rbf_test)

plt.rcParams['figure.dpi'] = 600
cm_test_disp_rbf.plot(colorbar=False, cmap='Oranges')
plt.title('SVM RBF: Teste')
plt.xlabel('Observado (Real)')
plt.ylabel('Classificado (Modelo)')
plt.show()

acc_rbf_test = accuracy_score(y_test, pred_rbf_test)
sens_rbf_test = recall_score(y_test, pred_rbf_test, pos_label=1)
espec_rbf_test = recall_score(y_test, pred_rbf_test, pos_label=0)
prec_rbf_test = precision_score(y_test, pred_rbf_test)

print("Avaliação do SVM RBF (Base de Teste)")
print(f"Acurácia: {acc_rbf_test:.1%}")
print(f"Sensibilidade: {sens_rbf_test:.1%}")
print(f"Especificidade: {espec_rbf_test:.1%}")
print(f"Precision: {prec_rbf_test:.1%}")

#%% Grid Search

# Especificar a lista de hiperparâmetros
param_grid = [
  {'C': [0.1, 1, 10], 'kernel': ['linear']},
  {'C': [0.1, 1, 10], 'degree': [2, 3], 'coef0': [0, 1], 'kernel': ['poly']},
  {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10, 100], 'kernel': ['rbf']},
]

# Identificar o algoritmo em uso
svm_grid = SVC()

# Treinar os modelos para o grid search
model_grid = GridSearchCV(estimator = svm_grid, 
                          param_grid = param_grid,
                          scoring='accuracy',
                          cv=5,
                          verbose=2)

model_grid.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
print(model_grid.best_params_)

# Gerando o modelo com os melhores hiperparâmetros
svm_best = model_grid.best_estimator_

# Valores preditos nas bases de treino e teste
pred_grid_train = svm_best.predict(X_train)
pred_grid_test = svm_best.predict(X_test)

#%% Análise pela matriz de confusão (base de treino)

cm_grid_train = confusion_matrix(pred_grid_train, y_train)
cm_train_disp_grid = ConfusionMatrixDisplay(cm_grid_train)

plt.rcParams['figure.dpi'] = 600
cm_train_disp_grid.plot(colorbar=False, cmap='Greens')
plt.title('SVM Grid: Treino')
plt.xlabel('Observado (Real)')
plt.ylabel('Classificado (Modelo)')
plt.show()

acc_grid_train = accuracy_score(y_train, pred_grid_train)
sens_grid_train = recall_score(y_train, pred_grid_train, pos_label=1)
espec_grid_train = recall_score(y_train, pred_grid_train, pos_label=0)
prec_grid_train = precision_score(y_train, pred_grid_train)

print("Avaliação do SVM Grid (Base de Treino)")
print(f"Acurácia: {acc_grid_train:.1%}")
print(f"Sensibilidade: {sens_grid_train:.1%}")
print(f"Especificidade: {espec_grid_train:.1%}")
print(f"Precision: {prec_grid_train:.1%}")

#%% Análise pela matriz de confusão (base de teste)

cm_grid_test = confusion_matrix(pred_grid_test, y_test)
cm_test_disp_grid = ConfusionMatrixDisplay(cm_grid_test)

plt.rcParams['figure.dpi'] = 600
cm_test_disp_grid.plot(colorbar=False, cmap='Greens')
plt.title('SVM Grid: Teste')
plt.xlabel('Observado (Real)')
plt.ylabel('Classificado (Modelo)')
plt.show()

acc_grid_test = accuracy_score(y_test, pred_grid_test)
sens_grid_test = recall_score(y_test, pred_grid_test, pos_label=1)
espec_grid_test = recall_score(y_test, pred_grid_test, pos_label=0)
prec_grid_test = precision_score(y_test, pred_grid_test)

print("Avaliação do SVM Grid (Base de Teste)")
print(f"Acurácia: {acc_grid_test:.1%}")
print(f"Sensibilidade: {sens_grid_test:.1%}")
print(f"Especificidade: {espec_grid_test:.1%}")
print(f"Precision: {prec_grid_test:.1%}")

#%% Importância das variáveis preditoras no modelo

result = permutation_importance(svm_best, X_test, y_test, 
                                n_repeats=10, 
                                scoring='accuracy',
                                random_state=100)

sort_importances = result.importances_mean.argsort()

importances = pd.DataFrame(result.importances[sort_importances].T,
                           columns=X_test.columns[sort_importances])

# Plotar no gráfico
plt.rcParams['figure.dpi'] = 600
importances.plot.box(vert=False, whis=10)
plt.title('Permutation Importances')
plt.xlabel('Redução na Acurácia')
plt.axvline(x=0, color='gray', linestyle=':')
plt.show()

#%% Realizando previsões para novas observações

# Note que as variáveis métricas são padronizadas
novo_cliente = pd.DataFrame({'idade': [((50-media_idade)/dp_idade)],
                             'acessos_mes': [((10-media_acessos)/dp_acessos)],
                             'valor_medio_transacao': [((3500-media_valor)/dp_valor)],
                             'sexo_masculino': [0],
                             'cidade_capital': [1],
                             'categoria_plano_basico': [1],
                             'categoria_plano_ouro': [0],
                             'categoria_plano_platinum': [0],
                             'categoria_plano_prata': [0],
                             'categoria_plano_sem': [0],
                             'reclamacao_sim': [1]})

# Previsão do modelo final
svm_best.predict(novo_cliente)

#%% Fim!