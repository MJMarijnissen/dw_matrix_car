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