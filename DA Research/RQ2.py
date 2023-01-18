from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import power_transform
import pandas as pd
import numpy as np
import pyreadstat
from scipy.stats import shapiro, boxcox, pearsonr, pointbiserialr, skew, kurtosis, mannwhitneyu
import matplotlib
import matplotlib.pyplot as plt
from math import log
from statsmodels.stats.stattools import durbin_watson
from math import log10
import seaborn

df, meta = pyreadstat.read_sav('PickUpLines.sav')

print(df.columns)

# spray = np.array(df[df['Scent'] == 1]['Receptivity']).reshape(-1, 1)
# nospray = np.array(df[df['Scent'] == 2]['Receptivity']).reshape(-1, 1)
# df['bc_rec'] = boxcox(df['Receptivity'], 0)
spray = df[df['Scent'] == 1]['Receptivity']
nospray = df[df['Scent'] == 2]['Receptivity']

indexAge = df[df['Receptivity'] == nospray.min()].index
df.drop(indexAge, inplace=True)

# seaborn.set(style='whitegrid')
# seaborn.boxplot(data=df, x='Scent', y='bc_rec')
# plt.show()

# print(len(spray), len(nospray))
# print(nospray.mean() - 3 * nospray.std(), nospray.min())

print(shapiro(spray), shapiro(nospray))

# plt.figure(figsize=(10, 7))
# n_bins = len(spray) // 2
# plt.hist(spray, bins=n_bins)
# plt.title('Receptivity cute-direct')
# plt.show()

# plt.figure(figsize=(10, 7))
# n_bins = len(nospray) // 2
# plt.hist(nospray, bins=n_bins)
# plt.title('Receptivity cute-direct')
# plt.show()

# print(skew(spray, axis=0, bias=True))
# print(skew(nospray, axis=0, bias=True))

log_spray, _ = boxcox(spray)
log_nospray, _ = boxcox(nospray)
print(shapiro(log_spray), shapiro(log_nospray))
# print(kurtosis(spray), kurtosis(nospray))

from scipy.stats import levene, ttest_ind, ttest_rel, ttest_1samp
lev = levene(np.array(log_spray), np.array(log_nospray), center='mean')
print(lev)

# print(spray.mean(), nospray.mean())
# stat, p_value = mannwhitneyu(spray, nospray, alternative="greater")
# print('Statistics=%.2f, p=%.2f' % (stat, p_value))
