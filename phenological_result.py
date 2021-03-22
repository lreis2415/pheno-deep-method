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
        
#Calculate RCLE value based on the time series of smoothed evi2 and lswi data at the cell location
# @evi2: the time series of evi2 at the cell location, as type of one-dimension np array
# @lswi: the time series of lswi at the cell location, as type of one-dimension np array
# @accumuDays: the days bewteen the date of each image and the earliest image,
#              as type of array, e.g. the accumuDays the this research is:
#              accumuDays=[0, 7, 20, 27, 47, 60, 68, 76, 81, 93, 108, 116, 128, 136, 141, 156, 173]
# return: RCLE value, LSWI_min
def calcRCLE(evi2, lswi, accumuDays):
    if validData(lswi)==True:
        heading = np.argmax(evi2)
        tillering = heading
        while(tillering > 0):
            tillering = tillering - 1
            days=accumuDays[heading]-accumuDays[tillering]
            if days>39 or tillering == 0:
                break
        evi2_interval = evi2[heading]-evi2[tillering]
        minLswi = np.min(lswi[tillering:heading+1])
        lswi_range = np.max(lswi[tillering:heading+1])-minLswi
        if abs(evi2_interval)>0:
            rcle = lswi_range/evi2_interval
        else:
            rcle = max_value
        return rcle, minLswi
    else:
        return float('nan'), float('nan')

# Create the distribution infomation of RCLE value, LSWI_min value, and classification results
# @EVI2_all: the time series of evi2 in rasters, 
#            as type of three-dimension np array [dim,height,width]
# @LSWI_all: the time series of lswi in rasters, 
#            as type of three-dimension np array [dim,height,width]
# @thres_rcle: threshold for RCLE value
# @thres_lm: threshold for lswi_min value
# @accumuDays: the days bewteen the date of each image and the earliest image,
#              as type of array, e.g. the accumuDays the this research is:
#              accumuDays=[0, 7, 20, 27, 47, 60, 68, 76, 81, 93, 108, 116, 128, 136, 141, 156, 173]
# @rlce_filename: the filename of the distribution information of stretchted RCLE values,
#                 values are in the range between 0-1, the file is in the format of .npy,
# @lswi_min_filename: the filename of the distribution information of LSWI_min values,
#                     in the format of .npy
# @classify_filename: the filename of the distribution information of classification results
#                     in the format of .npy
def map_phenology_results(EVI2_all, LSWI_all, thres_rcle, thres_lm, accumuDays, rlce_filename, lswi_min_filename, classify_filename):
    dim,Tif_height,Tif_width=EVI2_all.shape
    # calculate the RCLE and LSWI_min value at each cell location
    res = np.zeros((Tif_height,Tif_width))
    rcle_res = np.zeros((Tif_height,Tif_width))
    lswi_min_res = np.zeros((Tif_height,Tif_width))
    for j in range(Tif_height):
        for k in range(Tif_width):
            evi = EVI2_all[:,j,k]
            lswi = LSWI_all[:,j,k]
            rcle, lswi_min=calcRCLE(evi, lswi, accumuDays)
            rcle_res[j,k]=rcle
            lswi_min_res[j,k]=lswi_min
            if rcle< thres_rcle and lswi_min>thres_lm:
                res[j,k]=1
            else:
                res[j,k]=0
    np.save(lswi_min_filename, lswi_min_res)
    np.save(classify_filename, res)
    
    # stretch RCLE value to the range of 0-1
    freq = rcle_res.reshape((Tif_height*Tif_width))
    freq = freq[np.logical_not(np.isnan(freq))]
    freq = np.sort(freq)
    freq_shrink=np.zeros((10000))
    m=0
    length=float(freq.size)/10000.0
    for i in range(freq.size):
        if m==10000:
            break
        if i%int(length)==0:
            freq_shrink[m]=freq[i]
            m+=1
    if m<9999:
        freq_shrink[9999]=freq[freq.size-1]
    freq=freq_shrink
    freq_smaller_mid = freq[np.logical_not(freq>thres_rcle)]
    total_num_smaller = freq_smaller_mid.size
    freq_bigger_mid=freq[total_num_smaller:]
    total_num_bigger = freq_bigger_mid.size
    rcle_normal=np.zeros(rcle_res.shape)
    for i in range(Tif_height):
        for j in range(Tif_width):
            value = rcle_res[i,j]
            if math.isnan(value)==True:
                value_revise=float('nan')
            elif value>thres_rcle:
                pos=freq_bigger_mid[np.logical_not(freq_bigger_mid>value)]
                pos=pos.size
                value_revise = 0.5+0.5*(float(pos)/float(total_num_bigger))
            else:
                pos=freq_smaller_mid[np.logical_not(freq_smaller_mid>value)]
                pos=pos.size
                value_revise = 0.5*(float(pos)/float(total_num_smaller))
            rcle_normal[i,j]=value_revise
    np.save(rlce_filename, rcle_res)
    return

# example
EVI2_all = np.load(r'./EVI2_merge_filled.npy')
LSWI_all = np.load(r'./LSWI_merge_filled.npy')
rlce_filename = r'./RCLE.npy'
lswi_min_filename = r'./LSWI_min.npy'
classify_filename = r'./pheno_result.npy'
accumuDays=[0, 7, 20, 27, 47, 60, 68, 76, 81, 93, 108, 116, 128, 136, 141, 156, 173]
map_phenology_results(EVI2_all, LSWI_all, 0.9, 0.1, accumuDays,rlce_filename, lswi_min_filename, classify_filename)