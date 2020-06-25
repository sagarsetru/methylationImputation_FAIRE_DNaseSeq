import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def loadBedFile( baseDir, subBaseDir, fileName):
    "helper function to load bed files"
    return pd.read_csv(baseDir+subBaseDir+'/'+fileName, sep='\t', header=None)
#...

def getChromosomeIndices( bed, chromosome ):
    "return boolean for indices in bed file matching given chromosome"
    return [bed[0][x] == chromosome for x in range(len(bed[0]))]
#...

#def findOverlappingRegions (bed, openChromatinData ):

#set base directory for bed files
baseDir = '/Users/sagarsetru/Documents/Princeton/cos424/hw2/methylation_imputation/'

#load faire dataset NEW
print 'Loading faire and dnase seq data...'
faire_bed = loadBedFile(baseDir,'annotations','wgEncodeOpenChromFaireGm12878Pk.bed')
#load dnase seq dataset
ds_bed = loadBedFile(baseDir,'annotations','wgEncodeUWDukeDnaseGM12878.fdr01peaks.hg19.bed')


#chromosomes = ['chr1','chr2','chr6','chr7','chr11']
chromosomes = ['chr1']

for chrN in chromosomes:
    #load methylation data
    print 'Working on',chrN
    print 'Loading methylation data...'
    train_bed = loadBedFile(baseDir,'data','intersected_final_'+chrN+'_cutoff_20_train.bed')

    # get faire and DNase seq data from this chromosome
    faire_bed_chr = faire_bed[getChromosomeIndices(faire_bed,chrN)]
    ds_bed_chr = ds_bed[getChromosomeIndices(ds_bed,chrN)]
    # get start and end sites of open chromatin regions in array
    faire_sites = faire_bed_chr.loc[:,1:2].values
    ds_sites = ds_bed_chr.loc[:,1:2].values

    # get training sites in array
    train_data_sites = train_bed.loc[:,1:2].values

    #initialize arrays for faire, ds overlap
    faire_overlap = np.zeros(train_data_sites.shape[0])
    ds_overlap = np.zeros(train_data_sites.shape[0])

    #faire_sites_test = np.asarray([[10460, 10471],[10496, 10498]])
    #for (faire_start,faire_end) in faire_sites_test:
    print 'making faire overlap data'
    for (faire_start,faire_end) in faire_sites:
        #faire_start_diff = faire_start - train_data_sites[0:4,0]
        faire_start_diff = faire_start - train_data_sites[:,0]
        #print faire_start_diff
        #faire_end_diff = train_data_sites[0:4:,1] - faire_end
        faire_end_diff = train_data_sites[:,1] - faire_end
        #print faire_end_diff
        faire_start_info = np.where(faire_start_diff <= 0)
        #print faire_start_info
        faire_end_info = np.where(faire_end_diff <= 0)
        #print faire_end_info
        faire_overlap_inds = np.intersect1d(faire_start_info[0],faire_end_info[0])
        #print faire_overlap_inds
        faire_overlap[faire_overlap_inds]=1
    #...
    print 'making ds overlap data'
    for (ds_start,ds_end) in ds_sites:
        ds_start_diff = ds_start - train_data_sites[:,0]
        ds_end_diff = ds_end - train_data_sites[:,1]
        ds_start_info = np.where(ds_start_diff <= 0)
        ds_end_info = np.where(ds_end_diff <= 0)
        ds_overlap_inds = np.intersect1d(ds_start_info[0],ds_end_info[0])
        ds_overlap[ds_overlap_inds]=1
    #...

    # save overlap data to file
    saveOverlapDataDir = baseDir+'data'
    #if ~os.path.isdir(saveOverlapDataDir):
    #    os.makedirs(saveOverlapDataDir)
    #...
    os.chdir(saveOverlapDataDir)
    print 'saving ds and faire overlap data'
    #np.savetxt("faire_overlap_"+chrN+".csv",faire_overlap,delimiter=',')
    #np.savetxt("ds_overlap_"+chrN+".csv",ds_overlap,delimiter=',')

    
ds_overlap_pos = train_data_sites[ds_overlap_inds]
faire_overlap_pos = train_data_sites[faire_overlap_inds]
#np.savetxt("faire_overlap_pos_"+chrN+".csv",faire_overlap_pos,delimiter=',')
#np.savetxt("ds_overlap_"+chrN+".csv",ds_overlap_pos,delimiter=',')


#ds_plotData = np.concatenate((train_data_sites[:,0],ds_overlap),axis=1)
#plt.scatter(train_data_sites[:,0],ds_overlap)
N=100000 #good for faire
fc = np.convolve(faire_overlap, np.ones((N,))/N, mode='valid')
N=10000000
dc = np.convolve(ds_overlap, np.ones((N,))/N, mode='valid')

#plt.plot(np.convolve(faire_overlap, np.ones((N,))/N, mode='valid'))
#plt.xlabel('NaN CpG start sites, tissue index: '+str(x))
#plt.ylabel('Frequency')
#fileNameSave = 'NaN_CpGsites_tissueInd'+str(x)+'.png'
#plt.savefig(fileNameSave)

#faire_overlap_pos = train_data_sites[faire_overlap_inds]
#plt.plot(faire_overlap_pos)
