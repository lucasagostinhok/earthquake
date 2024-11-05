import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

data = pd.read_csv('Global_Earthquake_Data_sampled.csv')

data = data[['time', 'mag']]
data['time'] = pd.to_datetime(data['time'], errors='coerce')
data['mag'] = pd.to_numeric(data['mag'], errors='coerce') 
data = data.dropna(subset=['time', 'mag'])

data.set_index('time', inplace=True)
data = data.resample('M').mean()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(data.index, data['mag'], label='Magnitude Média dos Terremotos')
axes[0, 0].set_title('Magnitude Média Mensal de Terremotos')
axes[0, 0].set_xlabel('Ano')
axes[0, 0].set_ylabel('Magnitude')
axes[0, 0].legend()

plot_acf(data['mag'].dropna(), lags=24, ax=axes[0, 1])
axes[0, 1].set_title('Função de Autocorrelação (ACF)')
axes[0, 1].set_xlabel('Lags')
axes[0, 1].set_ylabel('Autocorrelação')

data['mag'].interpolate(method='linear', inplace=True)
decomposition = seasonal_decompose(data['mag'], model='additive', period=12)

decomposition.trend.plot(ax=axes[1, 0])
axes[1, 0].set_ylabel('Tendência')
axes[1, 0].set_title('Decomposição - Tendência')

decomposition.seasonal.plot(ax=axes[1, 1])
axes[1, 1].set_ylabel('Sazonalidade')
axes[1, 1].set_title('Decomposição - Sazonalidade')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
data['Moving_Avg'] = data['mag'].rolling(window=12).mean()
plt.plot(data.index, data['mag'], label='Magnitude Média dos Terremotos', alpha=0.6)
plt.plot(data.index, data['Moving_Avg'], color='red', label='Média Móvel (12 meses)')
plt.title('Magnitude Média Mensal de Terremotos com Média Móvel (12 meses)')
plt.xlabel('Ano')
plt.ylabel('Magnitude')
plt.legend()
plt.show()

adf_result = adfuller(data['mag'].dropna())
print("Teste Dickey-Fuller Aumentado (ADF):")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value}")
