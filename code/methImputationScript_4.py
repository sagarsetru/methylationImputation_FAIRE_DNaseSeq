# author Sagar Setru
# look at only 1 upstream and 1 downstream meth sites

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_squared_error

#import classifiers
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

#from sklearn.svm import SVR


def loadBedFile( baseDir, subBaseDir, fileName):
    "helper function to load bed files"
    return pd.read_csv(baseDir+subBaseDir+'/'+fileName, sep='\t', header=None)
#...

def makeDictCount( bed, columnInd ):
    "make dictionary from bed file counting number of rows having a given value for given column"
    bedDict = {}
    for x in bed[columnInd]:
        if x not in bedDict:
            bedDict[x] = 0
        #...
        bedDict[x] += 1
    #...
    return bedDict
#...

'''
def makeDictIndices( bed, columnInd ):
    "make dictionary from bed file with all row indices having a given value for given column"
    bedDict = {}
    counter = -1
    for x in bed[columnInd]:
        counter += 1
        if x not in bedDict:
            bedDict[x] = 0
        #...
        bedDict[x].append([counter])
    #...
    return bedDict
#...
'''

def getChromosomeIndices( bed, chromosome ):
    "return boolean for indices in bed file matching given chromosome"
    return [bed[0][x] == chromosome for x in range(len(bed[0]))]
#...

# http://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
def replaceNaNWithMean( x,a ):
    "replaces array x with mean along axis a"
    x_mean = stats.nanmean(x,axis=a)
    inds = np.where(np.isnan(x))
    x[inds]=np.take(x_mean,inds[1])
    return x

def runClassifier( Xtr, Ytr, Xte, classifier, *args ):
    "helper function to run different classifiers and get statistics on regression"
    "inputs: classifier, X, Y *args"
    "Xtr and Ytr are training data and labels"
    "Xte is testing data"
    "classifier is a string matching object name"
    "arguments after classifier are parameters for chosen classifier"
    if classifier is 'DecisionTreeClassifier':
        clf = DecisionTreeClassifier(max_depth = args[0])
        clf.fit(Xtr,Ytr)
        Y = clf.predict_proba(Xte)
        return Y
    #...
    if classifier is 'RandomForestClassifier':
        clf = RandomForestClassifier(n_estimators = args[0],max_features = args[1])
        clf.fit(Xtr,Ytr)
        #Y = clf.predict_proba(Xte)
        Y = clf.predict(Xte)
        return Y
    #...
    if classifier is 'LogisticRegression':
        clf = LogisticRegression(penalty = args[0])
        clf.fit(Xtr,Ytr)
        Y = clf.predict_proba(Xte)
        return Y
    #...
    if classifier is 'BernoulliNB':
        clf = BernoulliNB()
        clf.fit(Xtr,Ytr)
        Y = clf.predict_proba(Xte)
        return Y
    #...

    #return Y

#set base directory for bed files
baseDir = '/Users/sagarsetru/Documents/Princeton/cos424/hw2/methylation_imputation/'

#load faire dataset NEW
print 'Loading faire and dnase seq overlap data...'
#faire_bed = loadBedFile(baseDir,'annotations','wgEncodeOpenChromFaireGm12878Pk.bed')
#load dnase seq dataset
#ds_bed = loadBedFile(baseDir,'annotations','wgEncodeUWDukeDnaseGM12878.fdr01peaks.hg19.bed')


#chromosomes = ['chr1','chr2','chr6','chr7','chr11']
chromosomes = ['chr1']

showTime = 0;

for chrN in chromosomes:
    
    #load methylation data
    print 'Working on',chrN
    print 'Loading methylation data...'
    train_bed = loadBedFile(baseDir,'data','intersected_final_'+chrN+'_cutoff_20_train.bed')
    #test_bed = loadBedFile(baseDir,'data','intersected_final_'+chrN+'_cutoff_20_sample_partial.bed')
    ref_bed = loadBedFile(baseDir,'data','intersected_final_'+chrN+'_cutoff_20_sample_full.bed')
    print 'Loading faire and dnase seq overlap data...'
    ds_overlap = np.loadtxt(baseDir+'data/'+'ds_overlap_'+chrN+'.csv')
    faire_overlap = np.loadtxt(baseDir+'data/'+'faire_overlap_'+chrN+'.csv')

    # get training data in array
    train_data = train_bed.loc[:,4:36].values
    # get ref data into array
    ref_data = ref_bed.loc[:,4].values
    # get test data into array
    #test_data = test_bed.loc[:,4].values

    # get cpg start sites into array
    cpg_sites = ref_bed.loc[:,1].values
    
    # remove rows where ref data has NaN
    train_data = train_data[~np.isnan(ref_data)]
    ds_overlap = ds_overlap[~np.isnan(ref_data)]
    faire_overlap = faire_overlap[~np.isnan(ref_data)]
    cpg_sites = cpg_sites[~np.isnan(ref_data)]
    ref_data = ref_data[~np.isnan(ref_data)]

    # replace nan values in training data with mean at that CpG site
    train_data = replaceNaNWithMean(train_data,1)



    # binarize train data
    #train_meth_status = np.round(train_data)
    # binarize ref data
    #ref_meth_status = np.round(ref_data)
    ref_meth_status = ref_data
    ref_meth_status[ref_meth_status < np.mean(ref_data)] = 0
    ref_meth_status[ref_meth_status >= np.mean(ref_data)] = 1
    # initialize full feature vector, with ds and faire data
    fv_trainMethDsFaire = np.zeros((train_data.shape[0],train_data.shape[1]+2))
    # fill feature vector
    fv_trainMethDsFaire[:,0:train_data.shape[1]]=train_data
    fv_trainMethDsFaire[:,train_data.shape[1]]=ds_overlap
    fv_trainMethDsFaire[:,train_data.shape[1]+1]=faire_overlap

    # initialize full feature vector, with ds data
    fv_trainMethDs = np.zeros((train_data.shape[0],train_data.shape[1]+1))
    # fill feature vector
    fv_trainMethDs[:,0:train_data.shape[1]]=train_data
    fv_trainMethDs[:,train_data.shape[1]]=ds_overlap

    # initialize full feature vector, with ds and faire data
    fv_trainMethFaire = np.zeros((train_data.shape[0],train_data.shape[1]+1))
    # fill feature vector
    fv_trainMethFaire[:,0:train_data.shape[1]]=train_data
    fv_trainMethFaire[:,train_data.shape[1]]=faire_overlap

    dirSaves = (baseDir+'analysis/'+chrN+'/meth', baseDir+'analysis/'+chrN+'/meth_ds', baseDir+'analysis/'+chrN+'/meth_faire', baseDir+'analysis/'+chrN+'/meth_ds_faire') 
    #dirSaveBase = (baseDir+'analsysis/'+chrN+'/')
    #numNearestBases = np.asarray([100,150])

    timestr = time.strftime("%Y%m%d-%H%M%S")
    # choose n random indices for neighbor testing
    nTestingInds = 50
    #cpg_test_inds = np.round(np.random.uniform(np.max(numNearestBases),cpg_sites.shape[0]-np.max(numNearestBases),nTestingInds))
    cpg_test_inds = np.round(np.random.uniform(1,cpg_sites.shape[0]-1,nTestingInds))
    os.chdir(baseDir+'analysis/'+chrN)
    np.savetxt("testingInds_onlyNearestNeighbors"+timestr+".csv",cpg_test_inds,delimiter=',')
    #    np.savetxt("rmse_lrl1"+chrN+".csv",rmse_lrl1,delimiter=',')
    #testingWindow = np.asarray(100)
    counterFeatureSet = -1
    for featureSet in (train_data,fv_trainMethDs,fv_trainMethFaire,fv_trainMethDsFaire):
    #for featureSet in (fv_trainMethDsFaire,):
    #for featureSet in (train_data,):
        print 'Feature set: ',counterFeatureSet+1
        counterFeatureSet += 1


        pred_lrl1 = np.zeros(nTestingInds) 
        pred_lrl2 = np.zeros(nTestingInds) 
        pred_bnb = np.zeros(nTestingInds)
        pred_rf = np.zeros(nTestingInds)


        #nWindows = np.asarray(range(nNeighbors,cpg_sites.shape[0]-nNeighbors,testingWindow)).shape[0]

        r_lrl1 = np.zeros((nTestingInds,2))
        r_lrl2 = np.zeros((nTestingInds,2))
        r_bnb = np.zeros((nTestingInds,2))
        r_rf = np.zeros((nTestingInds,2))

        rmse_lrl1 = np.zeros(nTestingInds)
        rmse_lrl2 = np.zeros(nTestingInds)
        rmse_bnb = np.zeros(nTestingInds)
        rmse_rf = np.zeros(nTestingInds)

        print 'Running binary classifiers...'
        counterFold = -1
        for i in range(0,nTestingInds):
            counterFold += 1
            indTest = cpg_test_inds[counterFold]
            #cpg_site_start = cpg_sites[i,0]
            #cpg_site_end = cpg_sites[i,1]
            #cpg_site_ind1 = np.where(cpg_sites[:,0] < cpg_site_start)
            #cpg_site_ind2 = np.where(cpg_sites[:,1] > cpg_site_end)
            #print 'fold: ',counterFold+1
            # assign training and testing data
            Xtr = featureSet[indTest-1:indTest+2,:]
            # remove testing window
            Xtr1 = Xtr[:Xtr.shape[0]/2,:]
            Xtr2 = Xtr[Xtr.shape[0]/2+1:,:]
            Xtr = np.concatenate((Xtr1,Xtr2))
            #inds_te = np.asarray(range(i-testingWindow/2,+testingWindow/2))
            # Xte = featureSet[i,:].reshape(1,-1)
            Xte = featureSet[indTest,:].reshape(1,-1)
            Ytr = ref_meth_status[indTest-1:indTest+2]
            # remove testing window
            Ytr1 = Ytr[:Xtr.shape[0]/2]
            Ytr2 = Ytr[Xtr.shape[0]/2+1:]
            Ytr = np.concatenate((Ytr1,Ytr2))
            #Ytr = ref_data[i_tr]
            Yte = ref_data[indTest]

            # do classification
            start = time.time()
            #pred_lrl1[counterFold] = runClassifier( Xtr, Ytr, Xte, 'LogisticRegression', 'l1' )[:,1]
            lrl1t = time.time()
            #pred_lrl2[counterFold] = runClassifier( Xtr, Ytr, Xte, 'LogisticRegression', 'l2' )[:,1]
            lrl2t = time.time()
            #pred_bnb[counterFold] = runClassifier( Xtr, Ytr, Xte, 'BernoulliNB' )[:,1]
            bnbt = time.time()
            n_estimators = 100
            max_depth = 5
            pred_rf[counterFold] = runClassifier( Xtr, Ytr, Xte, 'RandomForestClassifier', n_estimators, max_depth )
            rft = time.time()
            if showTime == 1:
                print 'lrl1: ', lrl1t - start
                print 'lrl2: ', lrl2t - start
                print 'bnb: ', bnbt - start
                print 'rf: ', rft - start
            #...
            # determine statistics with this neighbor size (pearson r and RMSE)
            #Yte_all = ref_data[nNeighbors:cpg_sites.shape[0]-nNeighbors]
        Yte_all = ref_data[cpg_test_inds.astype(int)]
        #r_lrl1 = stats.pearsonr(pred_lrl1,Yte_all)
        #r_lrl2 = stats.pearsonr(pred_lrl2,Yte_all)
        #r_bnb = stats.pearsonr(pred_bnb,Yte_all)
        r_rf = stats.pearsonr(pred_rf,Yte_all)
        #rmse_lrl1 = np.sqrt(mean_squared_error(pred_lrl1,Yte_all))
        #rmse_lrl2 = np.sqrt(mean_squared_error(pred_lrl2,Yte_all))
        #rmse_bnb = np.sqrt(mean_squared_error(pred_bnb,Yte_all))
        rmse_rf = np.sqrt(mean_squared_error(pred_rf,Yte_all))

        #...

        #do random forest classification
        #maxLearners = 100
        #maxDepth = 5
        #rf = RandomForestClassifier(n_estimators = maxLearners, max_depth = maxDepth)
        #rf.fit(methDsFaire[0:500,1:],meth_b[0:500,0])
        #rf.fit(methDsFaire[0:500,1:],meth_b[0:500,0])
        #testArray[:,0] = meth[:,0]
        #testArray[:,1] = ds_overlap
        #testArray[:,2] = faire_overlap
        #pred_prob_rf=rf.predict_proba(methDsFaire[501,1:])
        #rf.fit(fv_methDsFaire[0:500],cpg_sites[0:500])
        #rf.fit(fv_trainMethDsFaire,ref_meth_status)
        #pred_prob_rf = rf.predict(fv_methDsFair[501])
        #pred_prob_rf = rf.predict_proba(fv_trainMethDsFaire[5001].reshape(1,-1))
        #pred_prob_rf = rf.predict_proba(ref_data[0:500].reshape(-1,1))

        #
        ### do analysis
        #move to analysis directory
        dirSave = dirSaves[counterFeatureSet]+'_singleSiteTests_onlyNextNeighbors'
        if not os.path.isdir(dirSave):
            os.makedirs(dirSave)
        #...
        os.chdir(dirSave)
        # save predictions
        #np.savetxt("pred_lrl1"+chrN+".csv",pred_lrl1,delimiter=',')
        #np.savetxt("pred_lrl2"+chrN+".csv",pred_lrl2,delimiter=',')
        #np.savetxt("pred_bnb"+chrN+".csv",pred_bnb,delimiter=',')
        np.savetxt("pred_rf"+chrN+".csv",pred_rf,delimiter=',')

        #np.savetxt("r_lrl1"+chrN+".csv",r_lrl1,delimiter=',')
        #np.savetxt("r_lrl2"+chrN+".csv",r_lrl2,delimiter=',')
        #np.savetxt("r_bnb"+chrN+".csv",r_bnb,delimiter=',')
        np.savetxt("r_rf"+chrN+".csv",r_rf,delimiter=',')

        #with open("rmse_lrl1"+chrN+".csv", 'w') as f:
        #    f.write('%d' % rmse_lrl1)
        #with open("rmse_lrl2"+chrN+".csv", 'w') as f:
        #    f.write('%d' % rmse_lrl2)
        #with open("rmse_bnb"+chrN+".csv", 'w') as f:
        #    f.write('%d' % rmse_bnb)
        with open("rmse_rf"+chrN+".csv", 'w') as f:
            f.write('%d' % rmse_rf)

        #np.savetxt("rmse_lrl1"+chrN+".csv",rmse_lrl1,delimiter=',')
        #np.savetxt("rmse_lrl2"+chrN+".csv",rmse_lrl2,delimiter=',')
        #np.savetxt("rmse_bnb"+chrN+".csv",rmse_bnb,delimiter=',')
        #np.savetxt("rmse_rf"+chrN+".csv",rmse_rf,delimiter=',')

    '''
    #meth value vs position plots
    methValVsPosition_saveDir = os.getcwd()+'/methValVsPosition'
    if not os.path.isdir(methValVsPosition_saveDir):
        os.makedirs(methValVsPosition_saveDir)
    #...
    for x in range(4,36):
        yVals = train_bed[x].values #meth values
        xVals = train_bed[1] #positions
        plt.plot(xVals,yVals)
        plt.xlabel('Methylation level, tissue index: '+str(x))
        plt.ylabel('Position on chromosome')
    #...
    '''

    '''
    #make histograms of CpG sites with NaN methylation levels
    nan_saveDir = os.getcwd()+'/NaN_CpGhists'
    if ~os.path.isdir(nan_saveDir):
        os.makedirs(nan_saveDir)
    #...
    os.chdir(nan_saveDir)
    for x in range(4,36):
        indexNan = train_bed[x].index[train_bed[x].apply(np.isnan)]
        plt.hist(train_bed[1][indexNan].tolist(),1000)
        plt.xlabel('NaN CpG start sites, tissue index: '+str(x))
        plt.ylabel('Frequency')
        fileNameSave = 'NaN_CpGsites_tissueInd'+str(x)+'.png'
        plt.savefig(fileNameSave)
        print 'saving',fileNameSave
    #...

    indexNan = test_bed[4].index[test_bed[4].apply(np.isnan)]
    plt.hist(test_bed[1][indexNan].tolist(),1000)
    plt.xlabel('NaN CpG start sites, test data')
    plt.ylabel('Frequency')
    fileNameSave = 'NaN_CpGsites_testData.png'
    plt.savefig(fileNameSave)
    print 'saving',fileNameSave

    indexNan = ref_bed[4].index[ref_bed[4].apply(np.isnan)]
    plt.hist(ref_bed[1][indexNan].tolist(),1000)
    plt.xlabel('NaN CpG start sites, ref data')
    plt.ylabel('Frequency')
    fileNameSave = 'NaN_CpGsites_refData.png'        
    plt.savefig(fileNameSave)
    print 'saving',fileNameSave
    '''


# list comprehension to check if methylation value in reference (actual value of NaN in the testing) exists in the training data
#test = [train_bed[x][0] == ref_bed[4][0] for x in range(37)]

# load given TF data from all cell types
#TFs_bed = loadBedFile(baseDir,'annotations/','wgEncodeRegTfbsClusteredV3.bed')
# load DNase 1 hypersensitivity data from all cell types
#DH_bed = loadBedFile(baseDir,'annotations/','wgEncodeRegDnaseClusteredV3.bed')






