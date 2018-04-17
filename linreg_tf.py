# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:30:56 2017

@author: DG
"""
import cx_Oracle as db
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

login = {'id':'user', 'pw':'abc123', 'host':'address', 'port':1521, 'sid':'xe'}
varResponse = ['RESPONSE']
reference = {'varTimeStart':'2017080100', 'varTimeEnd':'2017092400', 'varResponse':'RESPONSE'}
index = ('abcdefghijklmnopqrstuvwxyz')
thres = [.9]
epoch = 15
batch = 100
seed = 777

dsn = db.makedsn(login['host'], login['port'], login['sid'])
conn = db.connect(login['id'], login['pw'], dsn)

query = "SELECT DISTINCT datatype FROM dbtable WHERE datatype IS NOT NULL ORDER BY datatype"
cursor = conn.cursor()
cursor.execute(query)
result = cursor.fetchall()
listVar = [row[0] for row in result]

query = "SELECT column_name FROM user_tab_columns WHERE lower(table_name) = 'dbtable'"
cursor = conn.cursor()
cursor.execute(query)
result = cursor.fetchall()
listCol = [row[0] for row in result]

query = "SELECT * FROM dbtable WHERE datatype IS NOT NULL ORDER BY time, datatype"
cursor = conn.cursor()
cursor.execute(query)
result = cursor.fetchall()
cursor.close()

conn.close()

df = pd.DataFrame.from_records(result, columns=listCol)

for val in thres:
    dataRaw = pd.pivot_table(df, index="TIME", columns="DATATYPE", values="DATA")
    dataRaw.dropna(axis=1, thresh=(val * dataRaw.shape[0]), inplace=True)

    dataTime = dataRaw.index.tolist()
    #dataProcessed = dataRaw.interpolate(method='linear', limit=2, limit_direction='forward')
    dataProcessed = dataRaw.dropna(axis=0, how='any', inplace=False)
    
    tf.set_random_seed(seed)
    y_raw = pd.DataFrame.as_matrix(dataProcessed, columns=['RESPONSE'])
    x_raw = pd.DataFrame.as_matrix(dataProcessed.drop('RESPONSE', axis=1))
    y_scaler = preprocessing.StandardScaler().fit(y_raw)
    x_scaler = preprocessing.StandardScaler().fit(x_raw)
    y_data = y_scaler.transform(y_raw)
    x_data = x_scaler.transform(x_raw)

    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, dataProcessed.shape[1] - 1])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W = tf.Variable(tf.random_normal([dataProcessed.shape[1] - 1, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    # Hypothesis
    hypothesis = tf.matmul(X, W) + b
    
    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # Minimize
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train = optimizer.minimize(cost)
    
    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        #    mr = pd.DataFrame({'Step', 'Cost'})
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print({'Step':step, "Cost":cost_val})
    
    sess.close()
    
    print(hy_val)
