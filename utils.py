import os
import utils
import pandas as pd
import numpy  as np
#import matplotlib.pyplot as plt
#from scipy import stats
from pandas.tseries.offsets import *
#import statsmodels.formula.api as smf
from glob import glob
from datetime import datetime
import tensorflow as tf
import keras
from keras import Input, layers
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D, LeakyReLU, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
import random
#from sklearn.metrics import mean_squared_error as mse
#from scipy import stats
#from sklearn.linear_model import LinearRegression
#from fireTS.models import NARX
seed_value = 2020
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
from sklearn.preprocessing import QuantileTransformer 
normalizer = QuantileTransformer(output_distribution='normal')



#utils functions
def caricaLOB(stockType,perc=1):
    '''stockType== msft o bynd'''
    if stockType=='msft':
        lob_data=lob_data=pd.read_csv('/Users/macri/Desktop/MSFT_OF.csv')
        lob_data.drop(['idx'], axis=1,inplace=True)
        lob_data.reset_index(drop=True,inplace=True)
    elif stockType=='bynd':
        lob_data=pd.read_csv('/Users/macri/Desktop/BYND_OF.csv')
        lob_data.reset_index(drop=True,inplace=True)
    C= int(len(lob_data)*perc)
    b=lob_data.iloc[ :C].copy()
    ld=b.values.tolist()
    lob_data=b
    ld=lob_data.values.tolist()
    return lob_data


def midPrice(lob_data):
    a=lob_data['AP1'].values.tolist()
    b=lob_data['BP1'].values.tolist()
    m=np.zeros(len(a))
    for i in range(len(a)):
        m[i]=((a[i]/10**4)+(b[i]/10**4))/2
    return m

#def upDown(m):
#    u_d=np.zeros(len(a))
#    for i in range(1,len(a)):
#        if m[i-1]>m[i]:#scende il prezzo
#            u_d[i-1]=-1
#        elif m[i-1]<m[i]:#sale il prezzo
#            u_d[i-1]=1
#        else:
#            u_d[i-1]=0
#    return u_d


def makeDiff(m,d):
    ''' m= pd.DataFrame.m'''
    em=m
    dist=len(em)-d
    r=np.empty(dist)
    for i in range(dist):
        r[i]=np.log(em[i+d])-np.log(em[i])
    return r

def ret(m):
    r=np.zeros(len(m))
    for i in range(1,len(m)):
        r[i]=m[i-1]-m[i]
    return r

def deltaT(m):
    N=0
    d=np.zeros(len(m))
    for i in range(1,len(m)):
        d[i]=m[i-1]-m[i]
        if d[i]!=0:
            N+=1
    return (2.34*10**7)/N

def doHk(deltT,k):
    hk=np.zeros(k)
    for i in range(k):
        hk[i]=((1/5)*i*deltT)//1
    return hk

def retHk(m,hk):
    ret=np.zeros(100)
    for i,k in zip(range(len(m)),hk):
        ret[i]=m[i+k]-m[i]
    return ret

def OF_1(ld):
    of=np.zeros((len(ld),6))
    for i,ii in zip(range(1,12,2),range(0,6,2)): #giro sui prezzi ask e volumi ask di conseguenza
        for j in range(1,len(ld)): #giro sulle righe
            #ask
            if ld[j][i-1]>ld[j-1][i-1]: #p_t>p_t-1
                of[j-1][ii]=-1*ld[j][i]
            elif ld[j][i-1]<ld[j-1][i-1]: #p_t<p_t-1
                of[j-1][ii]=ld[j][i]
            elif ld[j][i-1]==ld[j-1][i-1]: #p_t=p_t-1
                of[j-1][ii]=ld[j][i]-ld[j-1][i]
    for w,ww in zip(range(3,12,4),range(1,7,2)):
        for jj in range(1,len(ld)):    
            #bid
            if ld[jj][w-1]>ld[jj-1][w-1]: #p_t>p_t-1
                of[jj-1][ww]=ld[jj][w]
            elif ld[jj][w-1]<ld[jj-1][w-1]: #p_t<p_t-1
                of[jj-1][ww]=-1*ld[jj][w]
            elif ld[jj][w-1]==ld[jj-1][w-1]: #p_t=p_t-1
                of[jj-1][ww]=ld[jj][w]-ld[jj-1][w]
    return of
    
def taglia_e_cuci(v):
    hi=np.quantile(v,0.05)
    lo=np.quantile(v, 0.95)
    b=np.clip(v, hi,lo)
    return b

def normTaglia(of_data):
    ''' prende un dataframe già predisposto: normalizza e taglia'''
    x1=of_data['aOF_1']
    x2=of_data['bOF_1']
    x3=of_data['aOF_2']
    x4=of_data['bOF_2']
    x5=of_data['aOF_3']
    x6=of_data['bOF_3']
    #x7=of_data['r'    ]
    aOF_1_per= normalizer.fit_transform((x1.values.reshape(-1,1))).flatten().tolist()#x1Per
    bOF_1_per= normalizer.fit_transform((x2.values.reshape(-1,1))).flatten().tolist()#x2Per
    aOF_2_per= normalizer.fit_transform((x3.values.reshape(-1,1))).flatten().tolist()#x3Per
    bOF_2_per= normalizer.fit_transform((x4.values.reshape(-1,1))).flatten().tolist()#x4Per
    aOF_3_per= normalizer.fit_transform((x5.values.reshape(-1,1))).flatten().tolist()#x5Per
    bOF_3_per= normalizer.fit_transform((x6.values.reshape(-1,1))).flatten().tolist()#x6Per
    #r_per    = normalizer.fit_transform((x7.values.reshape(-1,1))).flatten().tolist()#x7Per
    UPPERBOUND , LOWERBOUND =np.percentile(aOF_1_per, [0.005,99.995])
    UPPERBOUND1, LOWERBOUND1=np.percentile(bOF_1_per, [0.005,99.995])
    UPPERBOUND2, LOWERBOUND2=np.percentile(aOF_2_per, [0.005,99.995])
    UPPERBOUND3, LOWERBOUND3=np.percentile(bOF_2_per, [0.005,99.995])
    UPPERBOUND4, LOWERBOUND4=np.percentile(aOF_3_per, [0.005,99.995])
    UPPERBOUND5, LOWERBOUND5=np.percentile(bOF_3_per, [0.005,99.995])
    #UPPERBOUND6, LOWERBOUND6=np.percentile(r_per,     [0.005,99.995])
    x1Per=np.clip(aOF_1_per,UPPERBOUND ,LOWERBOUND )
    x2Per=np.clip(bOF_1_per,UPPERBOUND1,LOWERBOUND1)
    x3Per=np.clip(aOF_2_per,UPPERBOUND2,LOWERBOUND2)
    x4Per=np.clip(bOF_2_per,UPPERBOUND3,LOWERBOUND3)
    x5Per=np.clip(aOF_3_per,UPPERBOUND4,LOWERBOUND4)
    x6Per=np.clip(bOF_3_per,UPPERBOUND5,LOWERBOUND5)
    #x7Per=np.clip(r_per,UPPERBOUND6,LOWERBOUND6)




    data = {'aOF_1': x1Per,
            'bOF_1': x2Per,
            'aOF_2': x3Per,
            'bOF_2': x4Per,
            'aOF_3': x5Per,
            'bOF_3': x6Per,
            #'r'    : x7Per    
            }
    offlo=pd.DataFrame(data)
    return offlo


def prepXY(offlo,typo = 'cnnlstm'):
    ''' 
    da x e y 
    type = cnnlstm o altro
    '''
    ex = offlo.iloc[:,:-10].to_numpy()#[['aOF_1','bOF_1','aOF_2','bOF_2','aOF_3','bOF_3']]
    uai= offlo.iloc[:,-10:].to_numpy() #.pct_change().fillna(0)['r']
    shape = offlo.shape
    dimension=6
    lag = 100
    x=np.empty((shape[0]-lag, lag, dimension))
    y=np.empty((shape[0]-lag,10))
    for i in range(shape[0]-lag):
        x[i]=ex[i:i+lag]
        y[i]=uai[i+lag-1]
    if typo=='cnnlstm':
        X=x.reshape(-1,lag,shape[1]-10,1)
        Y=y.reshape(-1,10)
    else:
        X=x.reshape(-1,lag,shape[1]-10)
        Y=y.reshape(-1,10)
    return X,Y

def prepXY_cat(offlo,typo = 'cnnlstm'):
    ''' 
    da x e y 
    type = cnnlstm o altro
    '''


    ex = offlo[['aOF_1','bOF_1','aOF_2','bOF_2','aOF_3','bOF_3']].to_numpy()#
    uai= offlo['h'].to_numpy() #.pct_change().fillna(0)['r']
    shape = offlo.shape
    dimension=6
    lag = 100
    x=np.empty((shape[0]-lag, lag, dimension))
    y=np.empty((shape[0]-lag,2))
    for i in range(shape[0]-lag):
        x[i]=ex[i:i+lag]
        y[i]=uai[i+lag-1]

    #X=x.reshape(-1,lag,6,1)
    if typo=='cnnlstm':
        X=x.reshape(-1,lag,6,1)
        #Y=y.reshape(-1,1)
    else:
        X=x.reshape(-1,lag,6)        #Y=y.reshape(-1,1)


        #Y=y.reshape(-1,1)
    #y += 1 # relabel as 0, 1, 2

    Y = np_utils.to_categorical(y,2)
    return X,Y


def prepare_x_y(data, lag, dimension, typo="cnnlstm"):
    data = data
    shape = data.shape
    X = np.zeros((shape[0]-lag, lag, dimension))
    Y = np.zeros((shape[0]-lag,10))
    for i in range(shape[0]-lag):
        X[i] = data[i:i+lag, :dimension] # take the variables' columns as features
        Y[i] = data[i+lag-1, dimension:] # take the last columns as dep var
    if typo == "cnnlstm":
        X = X.reshape(X.shape[0], lag, dimension,1)
        Y = Y.reshape(Y.shape[0],10)
    else :
        X = X.reshape(X.shape[0], lag, dimension)
        Y = Y.reshape(Y.shape[0],10)
        
    return X,Y


def preparaRitorni( m, typo ='mstf'):
    df=pd.DataFrame(m['m'],columns=['r'])
    if typo =='msft':
        diff=[4  ,   8,    13,   17,   22,   26,   30,  35,   39, 44]#[44,  89, 133, 178, 223, 267, 312, 356, 401, 446]
        cols=["4",  "8",  "13", "17", "22", "26", "30", "35", "39" ,"44"]
    else:
        diff=[9,18,27,36,46,55,64,73,82,92]#[3,  6,  9, 12, 15, 18, 21, 24, 27, 30]#[31,  62,  93, 124, 155, 186, 217, 248, 279, 310]##[ 15,  30,  46,  61,  77,  92, 107, 123, 138, 154]#0.04
        cols=[ '38',  '77', '115', '154', '193', '231', '270', '308', '347', '386']
    for d,col in zip(diff,cols):
        df[col] = m['m'].pct_change(d)
    a=df[cols]
    return a.dropna()
#if __name__ == "__main__":
#    print("")
