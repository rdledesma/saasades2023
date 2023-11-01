#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:08:41 2023

@author: Rubén Darío Ledesma
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Metrics as ms

import seaborn as sns


#%%

d = pd.read_csv('measured/pay_60.csv')
c = pd.read_csv('cams/pay_60.csv', sep=";", header=42)


c = c.drop(['Reliability', '# Observation period','TOA'], axis=1)
d = pd.concat([d, c], axis=1)

plt.figure(1)
plt.plot(d.CTZ, d.ghi, '.r', markersize=1)
plt.xlabel("CTZ (rad)")
plt.ylabel("measured ghi")

plt.figure(2)
plt.plot(d.CTZ, d.GHI, '.r', markersize=1)
plt.xlabel("CTZ (rad)")
plt.ylabel("modeladed ghi CAMS")

plt.figure(3)
plt.plot(d.ghi, d.GHI, '.r', markersize=1)
plt.xlabel("measured ghi")
plt.ylabel("modeladed ghi CAMS")


#%%

true = d[d.alphaS>7].dropna().ghi
pred = d[d.alphaS>7].dropna().GHI

ms.r_mean_bias_error(true, pred)
ms.r_mean_absolute_error(true, pred)
ms.rrmsd(true, pred)




#%%



d['date'] = pd.to_datetime(d.date)

dna = d[d.date.dt.year == 2019]
dna = dna.dropna()
dna = dna[dna.alphaS>7]



plt.figure(4)
# Crear el scatter plot
sns.scatterplot(x='ghi', y='GHI', data=dna, color="black",  edgecolor='black')
# Añadir la recta de tendencia
slope, intercept = np.polyfit(dna['ghi'], dna['GHI'], 1)

sns.regplot(x='ghi', y='GHI', data=dna, scatter=False, color='red', label=f'Recta de tendencia: y = {slope:.2f}x + {intercept:.2f}')

# Añadir la recta x=y
plt.plot([dna['ghi'].min(), dna['ghi'].max()], [dna['ghi'].min(), dna['ghi'].max()], color='green', linestyle='--')

# Añadir etiquetas y título
plt.xlabel('ghi')
plt.ylabel('GHI')
plt.title('Gráfico de dispersión de ghi vs GHI')
plt.legend()
# Mostrar el gráfico
plt.show()


#%%







#%%




dna['kt'] = dna.ghi / dna['TOA']
dna['ktm'] = dna.GHI / dna['TOA']
dna['kcm'] = dna.GHI / dna['Clear sky GHI']

import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.glm(formula = "kt ~ ktm + kcm + alphaS", 
                data = dna, 
                family = sm.families.Binomial())
# Fit the model
result = model.fit()
predictions = dna.TOA * result.predict()


dna['ghi_lineal_rec'] = predictions

plt.figure(5)
# Crear el scatter plot
sns.scatterplot(x='ghi', y='ghi_lineal_rec', data=dna, color="black",  edgecolor='black')
# Añadir la recta de tendencia
slope, intercept = np.polyfit(dna['ghi'], dna['ghi_lineal_rec'], 1)

sns.regplot(x='ghi', y='ghi_lineal_rec', data=dna, scatter=False, color='red', label=f'Recta de tendencia: y = {slope:.2f}x + {intercept:.2f}')

# Añadir la recta x=y
plt.plot([dna['ghi'].min(), dna['ghi'].max()], [dna['ghi'].min(), dna['ghi'].max()], color='green', linestyle='--')

# Añadir etiquetas y título
plt.xlabel('ghi')
plt.ylabel('ghi_lineal_rec')
plt.title('Gráfico de dispersión de ghi vs ghi_lineal_rec')
plt.legend()
# Mostrar el gráfico
plt.show()


#%%


from scipy import interpolate
def ecdf(x): # empirical CDF computation
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def QuantileMappinBR(y_obs,y_mod): # Bias Removal using empirical quantile mapping
    y_cor = y_mod
    x_obs,cdf_obs = ecdf(y_obs)
    x_mod,cdf_mod = ecdf(y_mod)
    # Translate data to the quantile domain, apply the CDF operator
    cdf = interpolate.interp1d(x_mod,cdf_mod,kind='nearest',fill_value='extrapolate')
    qtile = cdf(y_mod)
    # Apply de CDF^-1 operator to reverse the operation to radiation domain
    cdfinv = interpolate.interp1d(cdf_obs,x_obs,kind='nearest',fill_value='extrapolate')
    y_cor = cdfinv(qtile)
    return y_cor

#%%


ghi_adapted = QuantileMappinBR(dna.ghi, dna.ghi_lineal_rec)
dna['ghi_adapted'] = ghi_adapted 


plt.figure(6)
# Crear el scatter plot
plt.plot(dna.index, dna.GHI, label="GHI cams")
plt.plot(dna.index, dna.ghi, label="GHI medida")
plt.plot(dna.index, dna.ghi_adapted,label="GHI adaptada")
plt.legend()

#%%

dtest = d
dtest = dtest.dropna()
dtest = dtest[dtest.alphaS>7]
dtest['kt'] = dtest.ghi / dtest['TOA']
dtest['ktm'] = dtest.GHI / dtest['TOA']
dtest['kcm'] = dtest.GHI / dtest['Clear sky GHI']


predictions = dtest.TOA * result.predict(dtest)
dtest['ghi_lineal_rec'] = predictions

ghi_adapted_test = QuantileMappinBR(dna.ghi, dtest.ghi_lineal_rec)
dtest['ghi_adapted'] = ghi_adapted_test


plt.figure(7)
# Crear el scatter plot
plt.plot(dtest.date, dtest.GHI, label="GHI cams")
plt.plot(dtest.date, dtest.ghi, label="GHI medida")
plt.plot(dtest.date, dtest.ghi_adapted, label="GHI adaptada")
plt.plot(dna.date, dna.ghi, label="GHI medida entrenamiento")
plt.legend()


#%%


plt.figure(8)
# Crear el scatter plot
sns.scatterplot(x='ghi', y='GHI', data=dtest, color="black",  edgecolor='black')
# Añadir la recta de tendencia
slope, intercept = np.polyfit(dtest['ghi'], dtest['GHI'], 1)

sns.regplot(x='ghi', y='GHI', data=dtest, scatter=False, color='red', label=f'Recta de tendencia: y = {slope:.2f}x + {intercept:.2f}')

# Añadir la recta x=y
plt.plot([dtest['ghi'].min(), dtest['ghi'].max()], [dtest['ghi'].min(), dtest['ghi'].max()], color='green', linestyle='--')

# Añadir etiquetas y título
plt.xlabel('ghi')
plt.ylabel('GHI Cams')
plt.title('Gráfico de dispersión de ghi vs GHI Cams')
plt.legend()
# Mostrar el gráfico
plt.show()


plt.figure(9)
# Crear el scatter plot
sns.scatterplot(x='ghi', y='ghi_adapted', data=dtest, color="black",  edgecolor='black')
# Añadir la recta de tendencia
slope, intercept = np.polyfit(dtest['ghi'], dtest['ghi_adapted'], 1)

sns.regplot(x='ghi', y='ghi_adapted', data=dtest, scatter=False, color='red', label=f'Recta de tendencia: y = {slope:.2f}x + {intercept:.2f}')

# Añadir la recta x=y
plt.plot([dtest['ghi'].min(), dtest['ghi'].max()], [dtest['ghi'].min(), dtest['ghi'].max()], color='green', linestyle='--')

# Añadir etiquetas y título
plt.xlabel('ghi')
plt.ylabel('GHI adaptada')
plt.title('Gráfico de dispersión de ghi vs GHI adaptada')
plt.legend()
# Mostrar el gráfico
plt.show()



#%%
true = dtest[dtest.alphaS>7].dropna().ghi
pred_cams = dtest[dtest.alphaS>7].dropna().GHI
pred_adap = dtest[dtest.alphaS>7].dropna().ghi_adapted



ms.r_mean_bias_error(true, pred_cams)
ms.r_mean_bias_error(true, pred_adap)

ms.r_mean_absolute_error(true, pred_cams)
ms.r_mean_absolute_error(true, pred_adap)

ms.rrmsd(true, pred_cams)
ms.rrmsd(true, pred_adap)



#%%
import numpy as np
from sklearn.linear_model import LinearRegression
"""
X variables regresoras
"""
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])


"""
variable objetivo
"""
y = np.dot(X, np.array([1, 2])) + 3


reg = LinearRegression().fit(X, y)
reg.score(X, y)

reg.coef_

reg.intercept_
reg.predict(np.array([[3, 5]]))



