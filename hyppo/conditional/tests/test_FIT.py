from operator import concat
from random import random

from numpy import floor
from numpy import np
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



def cross_validate(covarite,regressand):


def fit_test (x , y , z , n_perm =8 , frac_test=.1 ):
    n_samples=x.shape[0]
    n_test=floor(frac_test*n_samples)
    best_tree_x=cross_validate(concat(x,z),y)
    best_tree_nox=cross_validate(z,y)
    mses_x= list()
    mses_nox=list()
    for perm_id in range (n_perm):
        perm_ids=random.permuations(n_samples)
        x_test,x_train= x[perm_ids][:n_test],x[perm_id][n_test:]
        y_test, y_train = y[perm_ids][:n_test], y[perm_id][n_test:]
        z_test, z_train = z[perm_ids][:n_test], z[perm_id][n_test:]
        best_tree_x.train(concat(x_train,z_train),y_train)
        mses_x.append(
            mse(best_tree_x.predict(concat(x_test,z_test)),y_test))
        best_tree_nox.train(z_train,y_train)
        mses_nox.append((mse(best_tree_nox.predict(z_test))))

        t,pval=ttest_1samp(mses_nox-mses_x)
        if t<0:
            return 1- pval/2.
        else:
            return pval/2.