# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 01:24:03 2025

@author: João Mello
"""
import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split

import shap

df = pd.read_pickle('titanic1.pkl')

X = df.drop(columns='survived')
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=7)
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)

#%%

params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss",
}
model = xgboost.train(
    params,
    d_train,
    5000,
    evals=[(d_test, "test")],
    verbose_eval=100,
    early_stopping_rounds=20,
)

#%%
xgboost.plot_importance(model)
pl.title("xgboost.plot_importance(model)")
pl.show()



#%%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

#%%
grafico = shap.force_plot(explainer.expected_value, 
                          shap_values[0, :], 
                          X.iloc[0, :])

shap.save_html("force_plot_titanic.html", grafico)

#%%

grafico2 = shap.force_plot(explainer.expected_value, 
                           shap_values, 
                           X)
shap.save_html("force_plot2_titanic.html", grafico2)

#%%
shap.summary_plot(shap_values, X)
#%%
shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                     base_values=explainer.expected_value, 
                                     data=X.iloc[0], 
                                     feature_names=X.columns))
