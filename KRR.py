#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as MAE
from molml.features import Connectivity#, EncodedBond, CoulombMatrix
import statistics
from utils import load_qm7

if __name__ == "__main__":
    train_error = 0
    test_error= 0
    train_error_temp = [None] * 5
    test_error_temp = [None] * 5
    gamma = 1e-3
    alpha = 1e-7
    #kern = AtomKernel(gamma=gamma,transformer = Connectivity(n_jobs = -1), n_jobs=-1)
    #feat = CoulombMatrix(n_jobs = 1)
    #feats = [Connectivity(n_jobs = -1, depth = 3),
                 #EncodedBond(n_jobs = -1)]
    feat = (Connectivity(n_jobs = 1))
    #feat = (Connectivity(n_jobs = 1, use_coordination = True))

    #loop to  test each fold 
    for x in range(5):
       # Fit and transform test and train set
       Xin_train, Xin_test,y_train, y_test = load_qm7(x)
       #for feat in feats:
       
       #K_train = kern.fit_transform(Xin_train)
       #K_test = kern.transform(Xin_test)
       X_train = feat.fit_transform(Xin_train)
       X_test = feat.transform(Xin_test)
       clf = KernelRidge(alpha = alpha,gamma = gamma, kernel = "rbf")
       clf.fit(X_train, y_train)
       
       # Calculate train and test MAE
       train_error_temp[x] = MAE(clf.predict(X_train), y_train)
       test_error_temp[x] = MAE(clf.predict(X_test), y_test)
        
    print("Avg Train MAE: %.4f Avg Test MAE: %.4f" % 
          (statistics.mean(train_error_temp),statistics.mean(test_error_temp)))
    print()
    
    #caluate standard deviation
    print("Train Standard Deviation: %.4f, Test Standard Deviation: %.4f" % 
         ( statistics.pstdev(train_error_temp), statistics.pstdev(test_error_temp)))
    
