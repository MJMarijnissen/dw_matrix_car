# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:02:37 2020

@author: Kubus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_hdf("data/car.h5")
#print(df.shape)
#print(df.sample(5))

#peeking inside
print(df.columns.values)
print('\n')
print(df['price_value'].describe())
print('\n')
print(df['param_marka-pojazdu'].unique())
print('\n')


#extracting bar charts
df['price_value'].hist(bins=100)
df.groupby('param_marka-pojazdu')['price_value'].mean()
(
 df
 .groupby('param_marka-pojazdu')['price_value']
 .agg(np.median)
 .sort_values(ascending=False)
 .head(50)
 ).plot(kind='bar', figsize=(15,5))


#function for using in further analyzis
def group_and_barplot(feat_groupby, feat_agg = 'price_value', agg_funcs = [np.mean, np.median, np.size], feat_sort='mean', top=50, subplots=True):
    return (
        df
        .groupby(feat_groupby)[feat_agg]
        .agg(agg_funcs)
        .sort_values(by=feat_sort, ascending=False)
        .head(top)
        ).plot(kind='bar', figsize=(15,5), subplots = subplots);
