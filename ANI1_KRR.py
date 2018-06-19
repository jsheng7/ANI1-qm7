# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:12:53 2018

@author: rebeccasheng
"""
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as MAE
from molml.features import Connectivity, EncodedBond
import pickle
import numpy as np
import statistics

with open('gdict_dftb.p',"rb") as f:
    gdict = pickle.load(f, encoding = 'latin1')

train_set_size = 13200
test_set_size = 3300

#%% train set and test set of molecules
train_mols = [x for batch in gdict['train'] for x in batch]
test_mols = [x for batch in gdict['test'] for x in batch]

#%%atomization energy
train_energy = np.array([mol['Etot'] - mol['Edftb_elec'] - mol['Edftb_rep']for mol in train_mols])
train_energy = train_energy.astype(np.float32)
test_energy = np.array([mol['Etot'] - mol['Edftb_elec'] - mol['Edftb_rep'] for mol in test_mols])
test_energy = test_energy.astype(np.float32)

#%% atomic number
train_anum_temp = [mol['geom'].z for mol in train_mols]
anum_dict = {'1':'H', '6':'C', '8':'O' }
# Add padding for atomic number
train_anum = np.array(train_anum_temp)
for i in range(train_set_size):
     train_anum[i] = train_anum[i].astype(str)
     for j in range(len(train_anum[i])):
         train_anum[i][j] = anum_dict[train_anum[i][j]]

test_anum_temp = [mol['geom'].z for mol in test_mols]
test_anum = np.array(test_anum_temp)
for i in range(test_set_size):
     test_anum[i] = test_anum[i].astype(str)
     for j in range(len(test_anum[i])):
         test_anum[i][j] = anum_dict[test_anum[i][j]]

#%% cartesian coordinates 
train_coor_temp =[mol['geom'].rcart for mol in train_mols]
for i in range(train_set_size):
    train_coor_temp[i] = train_coor_temp[i].T
train_coor = np.asanyarray(train_coor_temp)
         
test_coor_temp = [mol['geom'].rcart for mol in test_mols]
for i in range(test_set_size):
    test_coor_temp[i] = test_coor_temp[i].T
test_coor = np.asanyarray(test_coor_temp)

#%%
#cross validation list
train_validation = ()
train_validation = np.arange(0,train_set_size, dtype = int).reshape(4,int(train_set_size/4))
np.random.shuffle(train_validation)
for i in range(4):
    np.random.shuffle(train_validation[i])
#train_validation = train_validation.tolist()

train_error = 0
test_error= 0
train_error_temp = [None] * 4
test_error_temp = [None] * 4

#feats  = [Connectivity(n_jobs = -1, depth = 2, use_bond_order = True),
                #EncodedBond(n_jobs = -1, smoothing = 'expit_pdf',max_depth = 1)]
#feat = Connectivity(n_jobs = -1, depth = 2, use_bond_order = True)
feat = Connectivity(n_jobs = -1)
alpha = 0.1
gamma = 1

for fold in range(4):
    train_folds = [x for x in range(4) if x != fold]
    train_idxs = np.ravel(train_validation[train_folds])
    test_idxs = np.ravel(train_validation[fold])
    
    Xin_train = list(zip(train_anum[train_idxs], train_coor[train_idxs]))
    Xin_test = list(zip(train_anum[test_idxs],train_coor[test_idxs]))
    y_train = train_energy[train_idxs]
    y_test = train_energy[test_idxs]
    
    #for feat in feats:
    X_train = feat.fit_transform(Xin_train)
    X_test = feat.transform(Xin_test)
    
    clf = KernelRidge(alpha = alpha,gamma = gamma, kernel = "rbf")
    clf.fit(X_train, y_train)
    train_error_temp[fold] = MAE(clf.predict(X_train), y_train)
    test_error_temp[fold] = MAE(clf.predict(X_test), y_test)
    
with open('ANI1_KRR.txt', 'a') as f:
    print('\n', feat)
    print("alpha: %.4f    gamma: %6f" % (alpha,  gamma))
    print("Avg Train MAE: %.6f Avg Test MAE: %.6f" % 
          (statistics.mean(train_error_temp),statistics.mean(test_error_temp)))
    print()
    #caluate standard deviation
    print("Train Standard Deviation: %.4f, Test Standard Deviation: %.4f" % 
          (statistics.pstdev(train_error_temp), statistics.pstdev(test_error_temp)))
'''
#%% Test section
finale_test_error = 0

Xin_train_final = list(zip(train_anum, train_coor))
Xin_test_final = list(zip(test_anum, test_coor))
y_train_final = train_energy
y_test_final = test_energy

for feat in feats:
    X_train_final = feat.fit_transform(Xin_train_final)
    X_test_final = feat.transform(Xin_test_final)

    clf = KernelRidge(alpha = alpha,gamma = gamma, kernel = "rbf")
    clf.fit(X_train, y_train)
train_error = MAE(clf.predict(X_train_final), y_train_final)
test_error = MAE(clf.predict(X_test_final), y_test_final)
with open('ANI1_LRR.txt', 'a') as f:
    print("Train MAE: %.6f    Test MAE: %.6f" % (train_error, test_error),file = f)
    '''