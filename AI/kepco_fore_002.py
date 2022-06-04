import pymysql

import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from glob import glob

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

import keras
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
import keras.backend as K 
from keras.callbacks import EarlyStopping

import tensorflow as tf
import pickle

import numpy as np
np.random.seed(42)

from datetime import datetime, timedelta
import time

def readData( sTmStt, sTmEnd ):
    cursor = data_db.cursor()
    sql = "SELECT mrdTm, validVol FROM naju_sum WHERE mrdTm >= '%s' AND mrdTm <= '%s'" % ( sTmStt, sTmEnd )
    cursor.execute( sql )
    res = cursor.fetchall()
    df = pd.DataFrame(res)
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m-%dT%H:%M:%SZ")
    #df["ds"] += pd.to_timedelta(9, unit='h')
    return df

def readWeather(sTmStt, sTmEnd):
    # humi 0 ~ 6, rain 0 ~ 65, temp -14.1 ~ 35.9, wind 0 ~ 7.1
    cursor = data_db.cursor()
    sql = "SELECT ymdt, temp, rain, wind, humi FROM naju_wthr WHERE ymdt >= '%s' AND ymdt <= '%s'" % (sTmStt, sTmEnd) 
    cursor.execute( sql )
    res = cursor.fetchall()
    df = pd.DataFrame(res)
    df.columns = ["ds", "temp", "rain", "wind", "humi"] # 
    df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m-%dT%H:%M:%SZ")
    #df["ds"] += pd.to_timedelta(9, unit='h')
    # df["humi"] = df["humi"] * 0.1
    df["day"] = df["ds"].apply(lambda v: setDayNight(v))
    df["week"] = df["ds"].apply(lambda v: setHolidays(v))
    return df

def setDayNight(v):
    if v.hour >= 7 and v.hour <= 18 :
        ret = 10
    else:
        ret = 0
    return ret

#def setDayNight( df ):
#    for i in df.index:
#        if df.loc[i, 'ds'].hour >= 7 and df.loc[i, 'ds'].hour <= 18 :
#            df.loc[i, 'day'] = 10
#        else:
#            df.loc[i, 'day'] = 0

def setHolidays( v ):
    if v.weekday() >= 5 :
        ret = 10
    else:
        ret = 0
    return ret

def mape_not_zero(y_true, y_pred):
    mape = 0
    for t, p in zip(y_true, y_pred):
        if t != 0:
            mape += np.abs((t - p) / t)
        elif p != 0:
            mape += np.abs((t - p) / p)

    mape /= len(y_true)
    return np.mean(mape) * 100

def rmse(y_true, y_pred):
    mse = (np.square(y_true - y_pred)).mean(axis=0)
    return np.sqrt(mse) 

#def setHolidays( df ):
#   for i in df.index:
#        if df.loc[i, 'ds'].weekday() >= 5 :
#            df.loc[i, 'week'] = 10
#        else:
#            df.loc[i, 'week'] = 0

print("keras version : ",keras.__version__)
print("tensorflow version : ", tf.__version__)

with open("model/model_rnn_kepco_001.pkl", "rb") as f:
    model = pickle.load(f)

n = 1
while( 1 ): 
    n=n+1
    time.sleep(1)
    print("==== test count : ", n)

    data_db = pymysql.connect(
        user='itman', 
        passwd='itman0808!', 
        host='192.168.0.105', 
        db='newkepco', 
        charset='utf8'
    )

    work_db = pymysql.connect(
        user='itman', 
        passwd='itman1234', 
        host='192.168.0.5', 
        db='car_chrg', 
        charset='utf8'
    )
    
    cursor = work_db.cursor()
    sql = "SELECT FORE_DT, WRK_STAT FROM CC_FORE_REQ " 
    cursor.execute( sql )
    res = cursor.fetchall()
    
    if not res:
        data_db.close()
        work_db.close()
        continue
    if res[0][1] != 'REQ':
        data_db.close()
        work_db.close()
        continue

    print("==== READ DATA...")
    cursor = work_db.cursor()
    cursor.execute( "UPDATE CC_FORE_REQ SET WRK_STAT = 'READ'" )
    cursor.execute( "COMMIT" )
        
    tmTestStt = datetime(res[0][0].year, res[0][0].month, res[0][0].day)
    tmTestStt -= pd.to_timedelta(2, unit='h')    
    tmTrainEnd = tmTestStt
    tmTrainEnd -= pd.to_timedelta(1, unit='h')
    print("=== test tm : ", tmTestStt, tmTrainEnd )    
    
    allX = readWeather("2017-01-04 00:00:00", "2018-12-27 23:59:59")
    allY = readData( "2017-01-04 00:00:00", "2018-12-27 23:59:59")
	
    print("==== CONVERT DATA...")
    cursor.execute( "UPDATE CC_FORE_REQ SET WRK_STAT = 'CONVERT'" )
    cursor.execute( "COMMIT" )
	
    for i in allX.index:
        if i >= 24:
            allX.loc[i, 'avg'] = allY.loc[i-24, 'y']
        else:
            allX.loc[i, 'avg'] = allY.loc[i, 'y']
        if i >= 48:
            allX.loc[i, 'avg2'] = allY.loc[i-48, 'y']
        else:
            allX.loc[i, 'avg2'] = allY.loc[i, 'y']
        
    allX.set_index('ds', inplace=True)
    allX.sort_values('ds', inplace=True)
    allY.set_index('ds', inplace=True)
    allY.sort_values('ds', inplace=True)

    # trainX = allX[:"2018-11-01 21:00:00"].copy()
    testX = allX[tmTestStt:].copy()
    # trainY = allY[:"2018-11-01 21:00:00"].copy()
    testY = allY[tmTestStt:].copy()        

    testsize = len(testY)
    timesteps = seq_length = 2 # 2시간을 학습으로.
    batch_size = 1
    data_dim = 8
    hidden_dim = 8
    output_dim = 1
    learing_rate = 0.0005
    iterations = 50000

    #scaler1 = RobustScaler()
    #scaler2 = RobustScaler()
    scaler3 = RobustScaler()
    scaler4 = RobustScaler()
    #trax = scaler1.fit_transform(trainX.values)
    #tray = scaler2.fit_transform(trainY.values)
    tstx = scaler3.fit_transform(testX.values)
    tsty = scaler4.fit_transform(testY.values)

    #dataX = []
    #dataY = []
    #for i in range(0, len(tray) - seq_length):
    #    _x = np.copy(trax[i:i + seq_length + 1])
    #    _y = [tray[i + seq_length]]
    #    dataX.append(_x)
    #    dataY.append(_y)
    #traX = np.array(dataX[:])
    #traY = np.array(dataY[:])[:,:,0]    
    
    data2X = []
    data2Y = []
    for i in range(0, len(tsty) - seq_length):
        _x = np.copy(tstx[i:i + seq_length + 1])
        _y = [tsty[i + seq_length]]
        data2X.append(_x)
        data2Y.append(_y)
    
    tstX = np.array(data2X[:])
    tstY = np.array(data2Y[:])[:,:,0]    
    print( "==== predict : ", tstX.shape, tstY.shape, len(tsty))

    print("==== FORECAST DATA...")
    cursor.execute( "UPDATE CC_FORE_REQ SET WRK_STAT = 'FORECAST'" )
    cursor.execute( "COMMIT" )

    predictions = model.predict(tstX, batch_size=1)
    preds = scaler4.inverse_transform(predictions)

    result = testY[-preds.size:].copy()
    result["yhat"] = preds[:]
    result

    testmape = mape_not_zero(result.y.values[:], result.yhat.values[:])
    testrmse = rmse(result.y.values[:], result.yhat.values[:])
    print("==== MAPE : ", testmape, testrmse)

    print("==== write forecast data...")
    cursor = work_db.cursor()
    
    for i in result.index:
        sql = " INSERT INTO CC_USE_VOL_FORE ( FORE_TM, FORE_VAL, REAL_VAL ) VALUES ( '%s', '%s', '%s' ) " \
            " ON DUPLICATE KEY UPDATE FORE_VAL=VALUES(FORE_VAL), REAL_VAL=VALUES(REAL_VAL) " \
            % ( i, result.loc[i,'yhat'], result.loc[i,'y'] ) 
        cursor.execute( sql )
        
    print("==== END!!!")
    cursor.execute( "UPDATE CC_FORE_REQ SET WRK_STAT = 'END'" )
    cursor.execute( "COMMIT" )

    data_db.close()
    work_db.close()
    
    print("==== work end !!!");
