# -*- coding: utf-8 -*-
import numpy as np
import math
import random
very_small=0.0000000000000001
max_value=30000
def validData(data):
    if np.count_nonzero(np.isnan(data))>10:
        return False
    else:
        return True
        

# random sampling
# for function getTrainSample
def getRandomPoint(num_point,realdata):
    dict = {}
    realdata_random = random.sample(realdata,num_point)
    for i in range(num_point):
        dict[i] = realdata_random[i]
    return dict

# get locations of rice paddy    
# for function getTrainSample
def getRiceData(rcle,lswi_min,thres, thres2):
    height,width = rcle.shape
    realdata = []
    for j in range(height):
        for k in range(width):
            if math.isnan(rcle[j,k]):
                continue
            elif rcle[j,k]<thres and rcle[j,k]>thres2 and  lswi_min[j,k]>0.1:
                realdata.append([j,k])
    return realdata

# get locations of non rice paddy
# for function getTrainSample
def getNonRiceData(rcle,lswi_min,thres, thres2):
    height,width = rcle.shape
    realdata = []
    for j in range(height):
        for k in range(width):
            if math.isnan(rcle[j,k]):
                continue
            elif rcle[j,k]>thres and rcle[j,k]<thres2:
                realdata.append([j,k])
    return realdata

# create sample with feature data
# for function getTrainSample
def getFeature(dict,sample_num,LSWI,EVI2,NDVI,blue,green,red,swir,nir):
    days = EVI2.shape[0]
    feature_sample = np.zeros((days*8,len(dict)))
    num=0
    for key, value in dict.items(): 
        feature_LSWI = LSWI[:,value[0],value[1]]
        is_valid=validData(feature_LSWI)
        if is_valid==True:
            feature = np.concatenate((feature_LSWI, EVI2[:,value[0],value[1]], NDVI[:,value[0],value[1]], blue[:,value[0],value[1]],green[:,value[0],value[1]], red[:,value[0],value[1]],swir[:,value[0],value[1]],nir[:,value[0],value[1]]))
            feature_sample[0:(days*8),num] = feature
            num+=1
    idx=np.random.randint(num, size=sample_num)
    return feature_sample[:,idx]

# get training samples with feature space based on phenological results
# @sample_num: the number of samples required
# @threshold: between 0 and 0.25, the threshold for defining sampling interval,
#             the interval for rice paddy sample would be (threshold,0.5-threshold),
#             the interval for non-rice paddy sample would be (0.5+threshold,1-threshold)
# @rcle: the stretched rcle distribution,
#        as type of two-dimension numpy array [height,width]
# @lswi_min: the stretched lswi_min distribution,
#            as type of two-dimension numpy array [height,width]
# @LSWI,EVI2,NDVI,blue,green,red,swir,nir: 
#            the time series of feature bands and indexes in rasters, 
#            as type of three-dimension np array [dim,height,width]
# return: sample for non-rice paddy with feature data, sample for rice paddy with feature data
def getTrainSample(sample_num, threshold,rcle,lswi_min,LSWI,EVI2,NDVI,blue,green,red,swir,nir):
    rice_realdata=getRiceData(rcle,lswi_min,0.5-threshold,threshold)
    nonrice_realdata=getNonRiceData(rcle,lswi_min,0.5+threshold,1-threshold)
    dict_0 = getRandomPoint(int(1.1*sample_num),nonrice_realdata)
    dict_1 = getRandomPoint(int(1.1*sample_num),rice_realdata)
    feature_sample0_train = getFeature(dict_0,sample_num,LSWI,EVI2,NDVI,blue,green,red,swir,nir)
    feature_sample1_train = getFeature(dict_1,sample_num,LSWI,EVI2,NDVI,blue,green,red,swir,nir)
    return feature_sample0_train,feature_sample1_train

    
# example
LSWI=np.load(r'./LSWI_merge_filled.npy')
EVI2=np.load(r'./EVI2_merge_filled.npy')
NDVI=np.load(r'./NDVI_merge_filled.npy')
blue=np.load(r'./blue_merge_filled.npy')
green=np.load(r'./green_merge_filled.npy')
red=np.load(r'./red_merge_filled.npy')
swir=np.load(r'./swir_merge_filled.npy')
nir=np.load(r'./nir_merge_filled.npy')

rcle=np.load(r'./RCLE.npy')
lswi_min=np.load(r'./LSWI_min.npy')
feature_sample0_train,feature_sample1_train=getTrainSample(350,0.2,rcle,lswi_min,LSWI,EVI2,NDVI,blue,green,red,swir,nir)
np.save(r'./sample0.npy', feature_sample0_train)
np.save(r'./sample1.npy', feature_sample1_train)