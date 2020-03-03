# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:38:40 2020

@author: Kubus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_hdf("data/car.h5")
print(df.shape)
print(df.sample(5))

df.columns.values
df['price_value'].hist(bins=100)
df['price_value'].describe()
df['param_marka-pojazdu'].unique()
df.groupby('param_marka-pojazdu')['price_value'].mean()

(
 df
 .groupby('param_marka-pojazdu')['price_value']
 .agg(np.median)
 .sort_values(ascending=False)
 .head(50)
 ).plot(kind='bar', figsize=(15,5))

(
 df
 .groupby('param_marka-pojazdu')['price_value']
 .agg([np.mean, np.median, np.size])
 .sort_values(by='mean', ascending=False)
 .head(50)
 ).plot(kind='bar', figsize=(15,5), subplots = True)