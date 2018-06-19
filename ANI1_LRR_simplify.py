# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:12:53 2018

@author: rebeccasheng
"""
from sklearn.linear_model import Ridge
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
#train_energy = np.array([mol['Etot'] for mol in train_mols])
train_energy = train_energy.astype(np.float32)
test_energy =np.array([mol['Etot'] - mol['Edftb_elec'] - mol['Edftb_rep']for mol in test_mols])
#test_energy =np.array([mol['Etot'] for mol in test_mols])
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

# run k-fold validation with maximum 2 features, with alpha range from 1 to 0.0001
def run_trial(feat1, output_file, feat2 = None, Gmax = 0):
#%%
    feats = {"1": Connectivity(n_jobs = -1),
                "1c": Connectivity(n_jobs = -1, use_coordination = True, depth = 1),
                "2": Connectivity(n_jobs = -1, depth = 2),
                "2b": Connectivity(n_jobs = -1, depth = 2, use_bond_order = True),
                "2NP": EncodedBond(n_jobs = -1, smoothing = 'norm',max_depth = Gmax),
                "2NC": EncodedBond(n_jobs = -1, smoothing = 'norm_cdf',max_depth = Gmax),
                "2LP": EncodedBond(n_jobs = -1, smoothing = 'expit_pdf',max_depth = Gmax),
                "2LC": EncodedBond(n_jobs = -1, smoothing = 'expit',max_depth = Gmax),
                "2SP": EncodedBond(n_jobs = -1, smoothing = 'spike',max_depth = Gmax),
                "2SC": EncodedBond(n_jobs = -1, smoothing = 'zero_one',max_depth = Gmax)}
    #cross validation list
    train_validation = ()
    train_validation = np.arange(0,train_set_size, dtype = int).reshape(4,int(train_set_size/4))
    np.random.shuffle(train_validation)
    for i in range(4):
        np.random.shuffle(train_validation[i])

    train_error_temp = [None] * 4
    test_error_temp = [None] * 4
    
    alpha_range = [1, 0, 0.1, 0.01, 0.001, 0.0001]
    
    # Perform k-fold validation. Can be made into a function
    for i, alpha in enumerate(alpha_range): 
        for fold in range(4):
            train_folds = [x for x in range(4) if x != fold]
            train_idxs = np.ravel(train_validation[train_folds])
            test_idxs = np.ravel(train_validation[fold])
        
            Xin_train = list(zip(train_anum[train_idxs], train_coor[train_idxs]))
            Xin_test = list(zip(train_anum[test_idxs],train_coor[test_idxs]))
            y_train = train_energy[train_idxs]
            y_test = train_energy[test_idxs]

            X_train1 = feats[feat1].fit_transform(Xin_train)
            X_test1 = feats[feat1].transform(Xin_test)
            if feat2 != None:
                X_train2 = feats[feat2].fit_transform(Xin_train)
                X_test2 = feats[feat2].transform(Xin_test)
                # concatenate feature vectors for combined features
                X_train = np.concatenate((X_train1, X_train2), axis = 1)
                X_test = np.concatenate((X_test1, X_test2), axis = 1)
            else:
                X_train = X_train1
                X_test = X_test1
            # LRR
            clf = Ridge(alpha = alpha)
            clf.fit(X_train, y_train)
    
            train_error_temp[fold] = MAE(clf.predict(X_train), y_train) * 627.509
            test_error_temp[fold] = MAE(clf.predict(X_test), y_test) * 627.509
    
        # Output the result to file
        with open(output_file, 'a') as f:
            print(feat1,file = f)
            if feat2 != None:
                print(feat2, file = f)
            print("Gmax = %d" % Gmax, file = f)
            print("alpha: %.4f" % alpha, file = f)
            print("Avg Train MAE: %.6f Avg Test MAE: %.6f" % 
              (statistics.mean(train_error_temp),statistics.mean(test_error_temp)), file = f)
            print()
            #caluate standard deviation
            print("Train Standard Deviation: %.4f, Test Standard Deviation: %.4f" % 
              (statistics.pstdev(train_error_temp), statistics.pstdev(test_error_temp)), file = f)

#%%
run_trial(feat1 = "1", feat2 = "2NP", Gmax = 3, output_file = "ANI1_LRR_2feats.txt")

#%% Test section
def run_test(feat1, output_file, alpha, feat2 = None, Gmax = 0): 
    feats = {"1": Connectivity(n_jobs = -1),
                "1c": Connectivity(n_jobs = -1, use_coordination = True, depth = 1),
                "2": Connectivity(n_jobs = -1, depth = 2),
                "2b": Connectivity(n_jobs = -1, depth = 2, use_bond_order = True),
                "2NP": EncodedBond(n_jobs = -1, smoothing = 'norm',max_depth = Gmax),
                "2NC": EncodedBond(n_jobs = -1, smoothing = 'norm_cdf',max_depth = Gmax),
                "2LP": EncodedBond(n_jobs = -1, smoothing = 'expit_pdf',max_depth = Gmax),
                "2LC": EncodedBond(n_jobs = -1, smoothing = 'expit',max_depth = Gmax),
                "2SP": EncodedBond(n_jobs = -1, smoothing = 'spike',max_depth = Gmax),
                "2SC": EncodedBond(n_jobs = -1, smoothing = 'zero_one',max_depth = Gmax)}
    
    Xin_train_final = list(zip(train_anum, train_coor))
    Xin_test_final = list(zip(test_anum, test_coor))
    y_train_final = train_energy
    y_test_final = test_energy
    
    X_train1 = feats[feat1].fit_transform(Xin_train_final)
    X_test1 = feats[feat1].transform(Xin_test_final)
    if feat2 != None:
        X_train2 = feats[feat2].fit_transform(Xin_train_final)
        X_test2 = feats[feat2].transform(Xin_test_final)
        X_train_final = np.concatenate((X_train1, X_train2), axis = 1)
        X_test_final = np.concatenate((X_test1, X_test2), axis = 1)
    else:
        X_train_final = X_train1
        X_test_final = X_test1
    clf = Ridge(alpha = alpha)
    clf.fit(X_train_final, y_train_final)
    train_error = MAE(clf.predict(X_train_final), y_train_final) * 627.509
    test_error = MAE(clf.predict(X_test_final), y_test_final) * 627.509
    with open(output_file, 'a') as f:
        print("Test set", file = f)
        print(feat1, file = f)
        if(feat2 != None):
            print(feat2, file = f)
        print("Gmax: %d" %Gmax)
        print("Train MAE: %.6f    Test MAE: %.6f" % (train_error, test_error),file = f)
        
#%%
#run_test(feat1 = "1", feat2 = "2NP", Gmax = 0, output_file = "ANI1_LRR_2feats_test.txt", alpha = 0.0001)
