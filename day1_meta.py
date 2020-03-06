# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:38:40 2020

@author: Kubus
"""

import pandas as pd
import numpy as np
import xgboost as xgb

from hyperopt import hp, fmin, tpe, STATUS_OK

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
        
#unfactorizing data to improve results (reduce error)
df['param_moc'] = df['param_moc'].map(lambda x: -1 if str(x) == 'None' else int(x.split(" ")[0]))
df['param_rok-produkcji'] = df['param_rok-produkcji'].map(lambda x: -1 if str(x) == 'None' else int(x))
df['param_pojemność-skokowa'] = df['param_pojemność-skokowa'].map(lambda x: -1 if str(x) == 'None' else int(x.split("cm")[0].replace(" ", "")))

def run_model(model, feats):
    X = df[feats].values
    Y = df['price_value'].values
    
    scores = cross_val_score(model, X, Y, cv=3, scoring = 'neg_mean_absolute_error')
    return np.mean(scores), np.std(scores)

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

def obj_func(params):
    print("training with params: ")
    print(params)
    
    mean_mae, score_std = run_model(xgb.XGBRegressor(**params), feats)
    return {'loss': np.abs(mean_mae), 'status': STATUS_OK}

#objective space
xgb_reg_params = {
    'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
    'max_depth': hp.choice('max_depth', np.arange(5,16,1, dtype=int)),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'seed': 0,
    }

#run
best = fmin(obj_func, xgb_reg_params, algo=tpe.suggest, max_evals=15)

best

