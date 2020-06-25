import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#import regression classifiers
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

a=1
b=2
c=3
d=4
def test_var_args(farg, *args):
    print "formal arg:", farg
    #for arg in args:
    #    print "another arg:", arg
    print args
    return args[0]+args[2]

def runRegressionClassifer( Xtr, Ytr, Xte, classifier, *args ):
    "helper function to run different classifiers and get statistics on regression"
    "inputs: classifier, X, Y *args"
    "Xtr and Ytr are training data and labels"
    "Xte is testing data"
    "classifier is a string matching object name"
    "arguments after classifier are parameters for chosen classifier"
    if classifier is 'DecisionTreeRegressor':
        regr = DecisionTreeRegressor(max_depth = args[0])
        regr.fit(Xtr,Ytr)
        Y = regr.predict(Xte)
    #...
    if classifier is 'RandomForestRegressor':
        print args[0]
        print args[1]
        regr = RandomForestRegressor(n_estimators = args[0],max_features = args[1])
        regr.fit(Xtr,Ytr)
        Y = regr.predict(Xte)
    #...
    if classifier is 'SVR':
        regr = SVR()
        regr.fit(Xtr,Ytr)
        Y = regr.predict(Xte)
    #...
    
    return Y
#...
# example from http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
rng = np.random.RandomState(1)
Xtr = np.sort(5 * rng.rand(80, 1), axis=0)
Ytr = np.sin(Xtr).ravel()
Ytr[::5] += 3 * (0.5 - rng.rand(16))
Xte = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
classifier = 'DecisionTreeRegressor'
classifier = 'RandomForestRegressor'
RFdict = {'RandomForestRegressor':(10,'auto')}
DT_maxDepth = 10
RF_nEstimators = 10
RF_depth = 'auto'
test = runRegressionClassifer( Xtr, Ytr, Xte, classifier, RFdict[classifier][0], RFdict[classifier][1] )


