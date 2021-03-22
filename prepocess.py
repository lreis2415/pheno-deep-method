import numpy as np
import math
import scipy as sp
import scipy.sparse
import scipy.linalg
from scipy.sparse.linalg import cg

very_small=0.0000000000000001    

def validData(data):
    if np.count_nonzero(np.isnan(data))>10:
        return False
    else:
        return True

def fillNan(data1):
    data=data1
    if validData(data)==False:
        for i in range(data.size):
            data[i]=float('nan')
        return data
    else:
        start=data[0]
        startLoc=0
        end=data[data.size-1]
        endLoc=data.size-1
        if math.isnan(start)==True:
            for i in range(data.size):
                if math.isnan(data[i])==False:
                    start=data[i]
                    startLoc=i
                    break
            for i in range(startLoc):
                data[i]=start
        if math.isnan(end)==True:
            for i in range(data.size-1,-1,-1):
                if math.isnan(data[i])==False:
                    end=data[i]
                    endLoc=i
                    break
            for i in range(endLoc+1,data.size,1):
                data[i]=end
        previousValue = start
        previousLoc = startLoc
        flag14=0
        value14=0
        for i in range(startLoc+1,endLoc+1,1):
            if math.isnan(data[i])==False:
                if i>previousLoc+1:
                    for m in range(i-previousLoc-1):
                        data[previousLoc+1+m] = data[previousLoc]+(data[i] - data[previousLoc])*float(accumuDays[previousLoc+m+1]-accumuDays[previousLoc])/float(accumuDays[i]-accumuDays[previousLoc])
                        if previousLoc+1+m==14:
                            flag14=1
                            value14=data[previousLoc+1+m]
                previousValue = data[i]
                previousLoc = i
        return data

def whitsm(y, lmda):
  m = len(y)
  E = sp.sparse.identity(m)
  d1 = -1 * np.ones((m),dtype='d')
  d2 = 3 * np.ones((m),dtype='d')
  d3 = -3 * np.ones((m),dtype='d')
  d4 = np.ones((m),dtype='d')
  D = sp.sparse.diags([d1,d2,d3,d4],[0,1,2,3], shape=(m-3, m), format="csr")
  z = sp.sparse.linalg.cg(E + lmda * (D.transpose()).dot(D), y)
  return z[0]

# time series of spectral bands and indices in the shape of [dim, height, width]
LSWI=np.load(r'./LSWI_merge.npy')
EVI2=np.load(r'./EVI2_merge.npy')
NDVI=np.load(r'./NDVI_merge.npy')
blue=np.load(r'./blue_merge.npy')
green=np.load(r'./green_merge.npy')
red=np.load(r'./red_merge.npy')
swir=np.load(r'./swir_merge.npy')
nir=np.load(r'./nir_merge.npy') 

dim,Tif_height,Tif_width=nir.shape
LSWI_filled=np.zeros((dim,Tif_height,Tif_width))
EVI2_filled=np.zeros((dim,Tif_height,Tif_width))
NDVI_filled=np.zeros((dim,Tif_height,Tif_width))
blue_filled=np.zeros((dim,Tif_height,Tif_width))
green_filled=np.zeros((dim,Tif_height,Tif_width))
red_filled=np.zeros((dim,Tif_height,Tif_width))
swir_filled=np.zeros((dim,Tif_height,Tif_width))
nir_filled=np.zeros((dim,Tif_height,Tif_width))
LSWI_smoothed=np.zeros((dim,Tif_height,Tif_width))
EVI2_smoothed=np.zeros((dim,Tif_height,Tif_width))


for i in range(Tif_height):
    if i%100==0:
        print(str(i)+" / "+str(Tif_height))
    for j in range(Tif_width):
        LSWI_filled[:,i,j]=fillNan(LSWI[:,i,j])
        EVI2_filled[:,i,j] = fillNan(EVI2[:,i,j])
        NDVI_filled[:,i,j] = fillNan(NDVI[:,i,j])
        blue_filled[:,i,j] = fillNan(blue[:,i,j])
        green_filled[:,i,j] = fillNan(green[:,i,j])
        red_filled[:,i,j] = fillNan(red[:,i,j])
        swir_filled[:,i,j] = fillNan(swir[:,i,j])
        nir_filled[:,i,j] = fillNan(nir[:,i,j])
        LSWI_smoothed[:,i,j] = whitsm(LSWI_filled[:,i,j],2)
        EVI2_smoothed[:,i,j] = whitsm(EVI2_filled[:,i,j],2)
       
np.save(r'./LSWI_merge_filled.npy',LSWI_filled)
np.save(r'./EVI2_merge_filled.npy',EVI2_filled)
np.save(r'./NDVI_merge_filled.npy',NDVI_filled)
np.save(r'./blue_merge_filled.npy',blue_filled)
np.save(r'./green_merge_filled.npy',green_filled)
np.save(r'./red_merge_filled.npy',red_filled)
np.save(r'./swir_merge_filled.npy',swir_filled)
np.save(r'./nir_merge_filled.npy',nir_filled)
np.save(r'./LSWI_merge_smoothed.npy',LSWI_smoothed)
np.save(r'./EVI2_merge_smoothed.npy',EVI2_smoothed)
