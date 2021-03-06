# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:58:50 2020

@author: Kubus
"""

import pandas as pd
import numpy as np
import xgboost as xgb
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score, KFold

import eli5
from eli5.sklearn import PermutationImportance

#read data
df = pd.read_hdf("data/car.h5")
print(df.shape)
print(df.sample(5))

#df=df[df['price_currency'] != 'EUR']

## Features
SUFFIX_CAT = "__cat"
for feat in df.columns:
    if isinstance(df[feat][0], list) :continue
    
    factorized_values = df[feat].factorize()[0]
    if SUFFIX_CAT in feat:
        df[feat] = factorized_values
    else:
        df[feat + SUFFIX_CAT] = factorized_values
        
#Decision tree
cat_feats = [x for x in df.columns if SUFFIX_CAT in x]
cat_feats = [x for x in cat_feats if 'price' not in x]
        
def run_model(model, feats):
    X = df[feats].values
    Y = df['price_value'].values
    
    scores = cross_val_score(model, X, Y, cv=3, scoring = 'neg_mean_absolute_error')
    return np.mean(scores), np.std(scores)

#Decision Tree
print("Decision Tree")
print(run_model(DecisionTreeRegressor(max_depth = 5), cat_feats))

#Random Forest
print("Random Forest")
print(run_model(RandomForestRegressor(max_depth = 5, n_estimators=50, random_state=0), cat_feats))

#XGBoost
xgb_params = {
    'max_depth': 5,
    'n_estimators': 50,
    #'learning_rate': 0.1,
    'seed': 0
    }

model = xgb.XGBRFRegressor(**xgb_params)
print("XGBoost")
print(run_model(model, cat_feats))

#most influential features
X = df[cat_feats].values
Y = df['price_value'].values
m = xgb.XGBRFRegressor(max_depth=5, n_estimators=50, learning_rate=0.1, seed=0)
m.fit(X,Y)

imp = PermutationImportance(m, random_state=0).fit(X,Y)
print(eli5.show_weights(imp, feature_names = cat_feats).data)
    
#from above code come out the most infuential features:
feats = ['param_napęd__cat',
'param_skrzynia-biegów__cat',
'param_faktura-vat__cat',
'param_rok-produkcji__cat',
'param_moc__cat',
'param_stan__cat',
'feature_kamera-cofania__cat',
'feature_łopatki-zmiany-biegów__cat',
'param_pojemność-skokowa__cat']

#unfactorizing data to improve results (reduce error)
df['param_moc'] = df['param_moc'].map(lambda x: -1 if str(x) == 'None' else int(x.split(" ")[0]))
df['param_rok-produkcji'] = df['param_rok-produkcji'].map(lambda x: -1 if str(x) == 'None' else int(x))
df['param_pojemność-skokowa'] = df['param_pojemność-skokowa'].map(lambda x: -1 if str(x) == 'None' else int(x.split("cm")[0].replace(" ", "")))

#creating feat list of most influential features
feats = ['param_napęd__cat',
'param_skrzynia-biegów__cat',
'param_faktura-vat__cat',
'param_rok-produkcji',
'param_moc',
'param_stan__cat',
'feature_kamera-cofania__cat',
'feature_łopatki-zmiany-biegów__cat',
'param_pojemność-skokowa']

#XGBoost most influantial
xgb_params = {
    'max_depth': 6,
    'n_estimators': 60,
    #'learning_rate': 0.1,
    'seed': 0
    }

model = xgb.XGBRFRegressor(**xgb_params)
print("XGBoost")
print(run_model(model, feats))