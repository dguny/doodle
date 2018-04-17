# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:23:09 2018

@author: DG
"""

import tkinter as tk
from tkinter import filedialog
import glob, os, re
import pandas as pd
from datetime import timedelta
from sqlalchemy import types, create_engine

db_info = {'user':'jack', 'pw':'jill', 'host':'beantree', 'port':999, 'sid':'orcl'}
col_rename = {'1':'a','2':'b','3':'c'}
col_save = ['a','b','g','h','i']
dtypes = {'a':int,'b':int,'c':int}

log_paths = []

root = tk.Tk()
root.withdraw()

folder_parent = filedialog.askdirectory(parent=root)
for folder_name in glob.glob(os.path.join(folder_parent,'?Pattern*','Folder')):
    for file_name in glob.glob(os.path.join(folder_parent,folder_name,'*.csv')):
        log_paths.append(file_name)

for i in range(len(log_paths)):
    log_paths[i] = log_paths[i].replace('\\','/')

cnt = 0

for path in log_paths:
    print(path)
    tmp = re.search('(\d)ID(\d{2})',path)
    if tmp.group(1) == '1':
        name = 'Jack'
    elif tmp.group(1) == '2':
        name = 'Jill'
    else:
        name = 'Giant'
    system = tmp.group(2)
    
    if cnt == 0:
        df = pd.read_csv(path,header=0,index_col=False,engine='python',parse_dates={'g':['date','time1'],'h':['date','time2'],'i':['date',' time3']})
        df['name'] = name
        df['system'] = system
        cnt += 1
    else:
        tmp_df = pd.read_csv(path,header=0,index_col=False,engine='python',parse_dates={'g':['date','time1'],'h':['date','time2'],'i':['date','time3']})
        tmp_df['name'] = name
        tmp_df['system'] = system
        df = pd.concat([df,tmp_df],ignore_index=True,copy=False)
        cnt += 1

# =============================================================================
# for non-CSV type text logs, use readlines
#     log = []
# 
#     with open(path) as f:
#         data = f.readlines()
#         
#     for record in data:
#         tmp = [name, system]
#         tmp.extend(record.split())
#         log.append(tmp)
# =============================================================================

df.rename(col_rename,axis='columns',inplace=True)
df = df.astype(dtypes)
df.loc[df.g > df.i,'g'] = df.loc[df.g > df.i,'g'] - timedelta(days=1)
df.loc[df.h > df.i,'h'] = df.loc[df.h > df.i,'h'] - timedelta(days=1)
df[['g','h','i']] = df[['g','h','i']].apply(pd.to_datetime, errors='coerce')

conn = create_engine('oracle+cx_oracle://%s:%s@%s:%i/%s' % (db_info['user'],db_info['pw'],db_info['host'],db_info['port'],db_info['sid']))
dtypes_sql = {c:types.VARCHAR(df[c].str.len().max()) for c in df.columns[df.dtypes == 'object'].tolist()}
df[col_save].to_sql('log_table', conn, if_exists='replace', index=False, dtype=dtypes_sql)
conn.close()
