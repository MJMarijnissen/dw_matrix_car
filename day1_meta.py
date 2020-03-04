# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:38:40 2020

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

print(mae(Y, y_pred))