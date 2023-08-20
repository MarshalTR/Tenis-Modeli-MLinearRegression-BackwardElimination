#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('tenis.csv')


#veri on isleme

#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
windyplay = veriler.iloc[:,3:5].apply(LabelEncoder().fit_transform)
veriler2 = pd.concat([veriler.iloc[:,0:3], windyplay], axis=1)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ColumnTransformer oluştur
ct = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['outlook'])],  # 'Outlook' sütununu dönüştür
    remainder='passthrough'  # Diğer sütunları passthrough olarak bırak
)

# Verileri dönüştür
sonveriler = pd.DataFrame(ct.fit_transform(veriler2))


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

print(y_pred)

import statsmodels.api as sm 

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

#sonveriler = sonveriler.iloc[:,1:]

import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)







