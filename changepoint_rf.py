# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:30:56 2017

@author: DG
"""
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import timeit
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

login = {'id':'user', 'pw':'abc123', 'host':'address', 'port':1521, 'sid':'xe'}
reference = {'varTimeStart':'2017080100', 'varTimeEnd':'2017092400', 'varResponse':'RESPONSE', 'varRemove':'CREATE_TIME', 'cycle':7, 'thres':.9}

conn = create_engine('oracle+cx_oracle://%s:%s@%s:%i/%s' % (login['id'],login['pw'],login['host'],login['port'],login['sid']))

query = "SELECT * FROM dbtable WHERE datatype IS NOT NULL ORDER BY time, datatype"
df = pd.read_sql(query, conn)

conn.close()

dataRaw = pd.pivot_table(df, index="TIME", columns="DATATYPE", values="DATA")
dataRaw.dropna(axis=1, thresh=(reference['thres'] * dataRaw.shape[0]), inplace=True)

#dataProcessed = dataRaw.interpolate(method='linear', limit=2, limit_direction='forward')
dataProcessed = dataRaw.dropna(axis=0, how='any', inplace=False)
dataTime = dataProcessed.index.tolist()

y_raw = dataProcessed[reference['varResponse']]
x_raw = dataProcessed.drop(reference['varResponse'], axis=1)

np.random.seed(777)
imp = pd.DataFrame(index=x_raw.columns)

def up(imp, x_raw, y_raw):
    start = 0
    mid = 0
    end = 1
        
    while end < len(dataTime):
        if (dataTime[-1].replace(hour=0, minute=0, second=0, microsecond=0) - dataTime[start].replace(hour=0, minute=0, second=0, microsecond=0)).days <= reference['cycle']:
            end = len(dataTime)
        else:
            for i in range(mid + 1, len(dataTime)):
                if (dataTime[i].replace(hour=0, minute=0, second=0, microsecond=0) - dataTime[start].replace(hour=0, minute=0, second=0, microsecond=0)).days >= 1:
                    mid = i
                    break
            
            for i in range(end + 1, len(dataTime)):
                if (dataTime[i].replace(hour=0, minute=0, second=0, microsecond=0) - dataTime[start].replace(hour=0, minute=0, second=0, microsecond=0)).days >= reference['cycle']:
                    end = i
                    break
            
        y = y_raw[start:end]
        x = x_raw[start:end]
        
        tic = timeit.default_timer()
        rf = RandomForestRegressor(n_estimators=10000, max_features='auto', max_leaf_nodes = int(.5 * x_raw.shape[1]), min_samples_leaf=.05, n_jobs=-1, random_state=None)
        rf.fit(x, y)
        #rf.predict([])
        
        temp = pd.DataFrame(rf.feature_importances_, index=x.columns)
        imp = pd.concat([imp, temp.rename({0: dataTime[end].strftime('%y%m%d')}, axis='columns')], axis=1, copy=False)
        toc = timeit.default_timer()
        print(toc - tic)
        
        start = mid
        
    return imp

def updown(imp, x_raw, y_raw):
    start = 0
    target = start + 1
    
    # initialization
    while target < len(dataTime):
        if (dataTime[target].replace(hour=0, minute=0, second=0, microsecond=0) - dataTime[0].replace(hour=0, minute=0, second=0, microsecond=0)).days < reference['cycle']:
            target += 1
        else:
            break
    
    pos = target
    end = pos + 1
    
    # main loop
    while end < len(dataTime):
        if (dataTime[-1].replace(hour=0, minute=0, second=0, microsecond=0) - dataTime[target].replace(hour=0, minute=0, second=0, microsecond=0)).days <= reference['cycle']:
            end = len(dataTime)
            for i in range(start, target):
                if (dataTime[target].replace(hour=0, minute=0, second=0, microsecond=0) - dataTime[i].replace(hour=0, minute=0, second=0, microsecond=0)).days <= reference['cycle']:
                    start = i
                    break
        else:
            if start > 0:
                for i in range(start, target):
                    if (dataTime[target].replace(hour=0, minute=0, second=0, microsecond=0) - dataTime[i].replace(hour=0, minute=0, second=0, microsecond=0)).days <= reference['cycle']:
                        start = i
                        break
            
            for i in range(end + 1, len(dataTime)):
                if (dataTime[i].replace(hour=0, minute=0, second=0, microsecond=0) - dataTime[target].replace(hour=0, minute=0, second=0, microsecond=0)).days >= reference['cycle']:
                    end = i
                    break
            
            for i in range(pos + 1, end):
                if (dataTime[i].replace(hour=0, minute=0, second=0, microsecond=0) - dataTime[target].replace(hour=0, minute=0, second=0, microsecond=0)).days == 1:
                    pos = i
                    break
        
        y = y_raw[start:end]
        x = x_raw[start:end]
        
        tic = timeit.default_timer()
        rf = RandomForestRegressor(n_estimators=10000, max_features='auto', max_leaf_nodes = int(.5 * x_raw.shape[1]), min_samples_leaf=.05, n_jobs=-1, random_state=None)
        rf.fit(x, y)
        #rf.predict([])
        
        temp = pd.DataFrame(rf.feature_importances_, index=x.columns)
        imp = pd.concat([imp, temp.rename({0: dataTime[target].strftime('%y%m%d')}, axis='columns')], axis=1, copy=False)
        toc = timeit.default_timer()
        print(toc - tic)
        
        target = pos
        start += 1
        
    return imp

#imp = updown(imp, x_raw, y_raw)
imp = up(imp, x_raw, y_raw)

writer = pd.ExcelWriter('changepoint_rf_%s.xlsx' % datetime.today().strftime('%y%m%d_%H%M'), engine='xlsxwriter', index_label='DATE')
imp.rename_axis('DATE', axis='columns').T.to_excel(writer, sheet_name='%s' % imp.columns[-1])
writer.save()
