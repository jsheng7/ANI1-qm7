# -*- coding: utf-8 -*-
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as MAE
from molml.features import Connectivity, EncodedBond
import statistics
import numpy as np
from utils import load_qm7

if __name__ == "__main__":
    train_error = 0
    test_error= 0
    train_error_temp = [None] * 5
    test_error_temp = [None] * 5
    #feat = CoulombMatrix(n_jobs = -1)
    #feats2 = [Connectivity(n_jobs = -1, depth = 2, use_bond_order = True),
                 #EncodedBond(n_jobs = -1, smoothing = 'expit_pdf',max_depth = 1)]
    #feat2 = Connectivity(n_jobs = -1, depth = 2)
    feat2b = Connectivity(n_jobs = -1, depth = 2, use_bond_order = True)
    #feat2s = [(Connectivity(n_jobs = -1, depth = 2)),
                    #EncodedBond(n_jobs = -1, max_depth = 1)]
    feat1 = Connectivity(n_jobs = 1)
    featLC  = EncodedBond(n_jobs = -1, smoothing = 'expit', max_depth = 1)
    featNP  = EncodedBond(n_jobs = -1, smoothing = 'norm', max_depth = 2)
    #feat1b = (Connectivity(n_jobs = 1, use_coordination = True))

    #loop to  test each fold 
    for x in range(5):
       # Fit and transform test and train set
       Xin_train, Xin_test,y_train, y_test = load_qm7(x)
       #for feat in featNP:
       #X_train = feat1.fit_transform(Xin_train)
       #X_test = feat1.transform(Xin_test)
       #for feat2 in feats2:
       X_train1 = feat1.fit_transform(Xin_train)
       X_test1 = feat1.transform(Xin_test)
       #X_train2b = feat2b.fit_transform(Xin_train)
       #X_test2b = feat2b.transform(Xin_test)
       #for feat in featsNP:
       X_trainLC = featLC.fit_transform(Xin_train)
       X_testLC = featLC.transform(Xin_test)
       X_train = np.concatenate((X_train1, X_trainLC), axis = 1)
       X_test = np.concatenate((X_test1, X_testLC), axis = 1)
       # Use Ridge linear regression
       clf = Ridge(alpha = 0.01)
       clf.fit(X_trainLC, y_train)
       # Calculate train and test MAE
       train_error_temp[x] = MAE(clf.predict(X_trainLC), y_train)
       test_error_temp[x] = MAE(clf.predict(X_testLC), y_test)
        
    print("Avg Train MAE: %.4f Avg Test MAE: %.4f" % 
          (statistics.mean(train_error_temp),statistics.mean(test_error_temp)))
    print()
    
    #caluate standard deviation
    print("Train Standard Deviation: %.4f, Test Standard Deviation: %.4f" % 
         ( statistics.pstdev(train_error_temp), statistics.pstdev(test_error_temp)))
    