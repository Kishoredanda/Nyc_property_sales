# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:51:40 2019

@author: 561719
"""

##########################Data Normalization#######################################################
import pandas as pd
import numpy as np


R1=pd.read_csv("C:\\Users\\561719\\Documents\\Imarticus_MLP\\NYC_property_sales\\nyc-rolling-sales.csv")
R2=R1.iloc[:,1:]
print(R2.head())

# Data Nromalization
R2.replace(to_replace=' -  ',value='NA',inplace=True)
R2.head()


sp=R2.loc[R2['SALE PRICE'] != 'NA']

sp1=sp.loc[R2['LAND SQUARE FEET'] != 'NA']

sp2=sp1.loc[R2['GROSS SQUARE FEET'] != 'NA']


sp2['SALE PRICE']=pd.to_numeric(sp2['SALE PRICE'])
sp2['LAND SQUARE FEET']=pd.to_numeric(sp2['LAND SQUARE FEET'])
sp2['GROSS SQUARE FEET']=pd.to_numeric(sp2['GROSS SQUARE FEET'])


mean_sp=int(sp2['SALE PRICE'].mean())
mean_lsq=int(sp2['LAND SQUARE FEET'].mean())
mean_gsq=int(sp2['GROSS SQUARE FEET'].mean())

R2['SALE PRICE'].replace(to_replace='NA',value=mean_sp,inplace=True)
R2['LAND SQUARE FEET'].replace(to_replace='NA',value=mean_lsq,inplace=True)
R2['GROSS SQUARE FEET'].replace(to_replace='NA',value=mean_gsq,inplace=True)
R2.dtypes

R2['SALE PRICE']=pd.to_numeric(R2['SALE PRICE'])
R2['LAND SQUARE FEET']=pd.to_numeric(R2['LAND SQUARE FEET'])
R2['GROSS SQUARE FEET']=pd.to_numeric(R2['GROSS SQUARE FEET'])
R2.dtypes

R2['SALE DATE']=pd.to_datetime(R2['SALE DATE'])
R2['SALE YEAR']=pd.DatetimeIndex(R2['SALE DATE']).year
R2.dtypes

R3=R2.drop(columns=['BLOCK','LOT','EASE-MENT','ADDRESS','APARTMENT NUMBER','ZIP CODE'],axis=1)
R3.dtypes

R3['Build_age']=abs(R3['SALE YEAR']-R3['YEAR BUILT'])
R3.dtypes

##########################Univariate Analsys#######################################################


build_age_boxplot_1=R3.boxplot(column=['Build_age'],figsize=(12,8))

#R4=R3.loc[R3['YEAR BUILT'] != 0]

Q1=R3['Build_age'].quantile(0.25)
print(Q1)
Q3=R3['Build_age'].quantile(0.75)
print(Q3)
IQR=Q3-Q1
print(IQR)

R4=R3[~((R3['Build_age']<(Q1 - 1.5 * IQR)) | (R3['Build_age']>(Q3 + 1.5 * IQR)))]
R4.shape


build_age_boxplot_2=R4.boxplot(column=['Build_age'],figsize=(12,8))

R4.dtypes
R5=R4.drop(columns=['SALE YEAR','SALE DATE','YEAR BUILT'])


R5['Build_age'].plot.hist()
R5.dtypes
R5['NEIGHBORHOOD']=R5['NEIGHBORHOOD'].astype('category')
R5['BUILDING CLASS CATEGORY']=R5['BUILDING CLASS CATEGORY'].astype('category')
R5['neigh_cat']=R5['NEIGHBORHOOD'].cat.codes
R5['build_cat']=R5['BUILDING CLASS CATEGORY'].cat.codes
R5.dtypes
R5=R5.rename(columns={'SALE PRICE':'sale_price','LAND SQUARE FEET':'land_sft','GROSS SQUARE FEET':'gross_sft'
                      ,'TOTAL UNITS':'total_units','TAX CLASS AT TIME OF SALE':'tax_at_sale'})
R5.plot(kind='scatter',x='total_units',y='sale_price',alpha=0.2,figsize=(12,8))
total_units_boxplot_1=R5.boxplot(column=['total_units'],figsize=(12,12))


import matplotlib.pyplot as plt
R5.hist(bins=50,figsize=(20,15))
plt.show()

corr_matrix=R5.corr()
corr_matrix['sale_price'].sort_values(ascending=False)

R5.plot(kind='scatter',x='gross_sft',y='sale_price',alpha=0.5)
R5['price_per_sft']=R5['sale_price']/R5['gross_sft']
corr_matrix=R5.corr()
corr_matrix['sale_price'].sort_values(ascending=False)
R5.dtypes
R5.shape
price_per_sft_boxplot_1=R5.boxplot(column=['price_per_sft'],figsize=(12,8))
#R5.replace([np.inf,-np.inf],np.nan)
R6=R5.replace(to_replace=np.inf,value=np.nan)
R6.dropna(subset=['price_per_sft'],inplace=True)
R6.shape
R6.to_csv("C:\\Users\\561719\\Documents\\Imarticus_MLP\\NYC_property_sales\\nyc-rolling-sales_R6.csv",index=False)
corr_matrix=R6.corr()
corr_matrix['sale_price'].sort_values(ascending=False)

##########################OLS Linear Regression#######################################################
import statsmodels.api as sm
import statsmodels.formula.api as smf

#results=smf.ols('sale_price ~ Build_age + land_sft + gross_sft + neigh_cat + build_cat + BOROUGH',data=R5).fit()
#print(results.summary())
#results=smf.ols('sale_price ~ Build_age + land_sft + gross_sft + build_cat + BOROUGH',data=R5).fit()
#print(results.summary())
#results=smf.ols('sale_price ~ Build_age + land_sft + gross_sft + build_cat + total_units + tax_at_sale',data=R5).fit()
#print(results.summary())
#results=smf.ols('sale_price ~ land_sft + gross_sft + build_cat + total_units + tax_at_sale',data=R5).fit()
#print(results.summary())
results=smf.ols('sale_price ~ price_per_sft + gross_sft + total_units',data=R6).fit()
print(results.summary())

X=R6[['price_per_sft','gross_sft','total_units']]
Y=R6['sale_price']

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
print(X_test.head())
Y_pred=reg.predict(X_test)
print(Y_pred)
reg_score=reg.score(X_test,Y_test)
print("Linear Regression R Squared: %.4f" % reg_score)

from sklearn.metrics import mean_squared_error
line_mse=mean_squared_error(Y_pred,Y_test)
line_rmse=np.sqrt(line_mse)
print("Linear Regression RSME: %.4f" % line_rmse)

##################################Random Forest############################################################
from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor(random_state=40)
forest_reg.fit(X_train,Y_train)
forest_score=forest_reg.score(X_test,Y_test)
print("Random Forest R Squared: %.4f" % forest_score)
Y_pred=forest_reg.predict(X_test)
forest_mse=mean_squared_error(Y_pred,Y_test)
forest_rmse=np.sqrt(forest_mse)
print("Random Forest RSME: %.4f" % forest_rmse)















































