import cx_Oracle as db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model

login = {'id':'user', 'pw':'abc123', 'host':'address', 'port':1521, 'sid':'xe'}
varResponse = ['RESPONSE']
reference = {'varTimeStart':'2017080100', 'varTimeEnd':'2017090100', 'varResponse':'RESPONSE'}
thres = .9
epoch = 15
batch = 100

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

dataRaw = pd.pivot_table(df, index="TIME", columns="DATATYPE", values="DATA")
dataRaw.dropna(axis=1, thresh=(thres * dataRaw.shape[0]), inplace=True)

#dataProcessed = dataRaw.interpolate(method='linear', limit=2, limit_direction='forward')
dataProcessed = dataRaw.dropna(axis=0, how='any', inplace=False)
dataTime = dataProcessed.index.tolist()

y_raw = pd.DataFrame.as_matrix(dataProcessed, columns=[reference['varResponse']])
x_raw = pd.DataFrame.as_matrix(dataProcessed.drop(reference['varResponse'], axis=1))
y_scaler = preprocessing.StandardScaler().fit(y_raw)
x_scaler = preprocessing.StandardScaler().fit(x_raw)
y_data = y_scaler.transform(y_raw)
x_data = x_scaler.transform(x_raw)

np.random.seed(777)
n_alphas = 2000
alphas = np.logspace(-10, -2, n_alphas)
coefs = []
alpha = 4e-4

for val in alphas:
   lasso_reg = linear_model.Lasso(alpha=val, fit_intercept=False, normalize = False)
   lasso_reg.fit(x_data, y_data)
   coefs.append(lasso_reg.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
