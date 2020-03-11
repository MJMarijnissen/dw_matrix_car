# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:44:21 2020

@author: Kubus
"""


def run_model(model, feats):
    X = df[feats].values
    Y = df['price_value'].values
    
    scores = cross_val_score(model, X, Y, cv=3, scoring = 'neg_mean_absolute_error')
    return np.mean(scores), np.std(scores)

#function for using in further analyzis
def group_and_barplot(feat_groupby, 
                      feat_agg = 'price_value', 
                      agg_funcs = [np.mean, np.median, np.size], 
                      feat_sort='mean', 
                      top=50, 
                      subplots=True
                      ):
    return (
        df
        .groupby(feat_groupby)[feat_agg]
        .agg(agg_funcs)
        .sort_values(by=feat_sort, ascending=False)
        .head(top)
        ).plot(kind='bar', figsize=(15,5), subplots = subplots);

def features_manip(F):
    SUFFIX_CAT = "__cat"
    for feat in F.columns:
        if isinstance(F[feat][0], list) :continue
    
        factorized_values = F[feat].factorize()[0]
        if SUFFIX_CAT in feat:
            F[feat] = factorized_values
        else:
            F[feat + SUFFIX_CAT] = factorized_values
    return F