# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:55:09 2020

@author: Kubus
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score

import eli5
from eli5.sklearn import PermutationImportance

#read data
df = pd.read_hdf("data/car.h5")
print(df.shape)
print(df.sample(5))

df=df[df['price_currency'] != 'EUR']

#Dummy model

feats = ['car_id']
X = df[ feats ].values
Y = df['price_value'].values

model = DummyRegressor()
model.fit(X,Y)
y_pred = model.predict(X)

print("Dummy model error: ", mae(Y, y_pred))

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
        
X = df[cat_feats].values
Y = df['price_value'].values

model = DecisionTreeRegressor(max_depth = 5)

scores = cross_val_score(model, X, Y, cv=3, scoring = 'neg_mean_absolute_error')
print("Decision tree error: ",np.mean(scores))

#most influential features
m = DecisionTreeRegressor(max_depth=5)
m.fit(X,Y)

imp = PermutationImportance(m, random_state=0).fit(X,Y)
print(eli5.show_weights(imp, feature_names = cat_feats).data)
