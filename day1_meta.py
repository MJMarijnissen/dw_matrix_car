# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:38:40 2020

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

# #Decision Tree
# print("Decision Tree")
# print(run_model(DecisionTreeRegressor(max_depth = 5), cat_feats))

# #Random Forest
# print("Random Forest")
# print(run_model(RandomForestRegressor(max_depth = 5, n_estimators=50, random_state=0), cat_feats))

#XGBoost
xgb_params = {
    'max_depth': 5,
    'n_estimators': 50,
    'learning_rate': 0.1,
    'seed': 0
    }

model = xgb.XGBRFRegressor(**xgb_params)
print("XGBoost")
print(run_model(model, cat_feats))

#most influential features
# m = DecisionTreeRegressor(max_depth=5)
# m.fit(X,Y)

# imp = PermutationImportance(m, random_state=0).fit(X,Y)
# print(eli5.show_weights(imp, feature_names = cat_feats).data)
