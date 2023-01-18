from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import power_transform
import pandas as pd
import numpy as np
import pyreadstat
from scipy.stats import shapiro, boxcox, pearsonr, pointbiserialr, mannwhitneyu, chisquare, contingency, chi2_contingency
import matplotlib
import matplotlib.pyplot as plt
from math import log
from statsmodels.stats.stattools import durbin_watson
from math import log10

df, meta = pyreadstat.read_sav('PickUpLines.sav')

print(df.columns)
# print(meta.column_labels)
# print(df.PickUp.value_counts())
# print(meta.variable_value_labels)
# print(df.Receptivity.value_counts())

# plt.scatter(df['Receptivity'], df['PickUp'], color='b')
# plt.xlabel("Receptivity")
# plt.ylabel("PickUp")
# plt.show()

# print(len(df.ParticipantNumber), len(df[df['PickUp'] == 1]), len(df[df['PickUp'] == 2]))
# print(df.ParticipantNumber)

# direct = np.array(df[df['PickUp'] == 2]['Receptivity']).reshape(-1, 1)
# cute = np.array(df[df['PickUp'] == 1]['Receptivity']).reshape(-1, 1)
# df['bc_rec'], _ = boxcox(df['Receptivity'])
# df['bc_rec'] = np.log(df['Receptivity'])
# for i, row in df.iterrows():
#     # print(row)
#     row['Receptivity'] = 1 / row['Receptivity']
direct = df[df['PickUp'] == 2]['Receptivity']
cute = df[df['PickUp'] == 1]['Receptivity']
print(shapiro(direct))
print(shapiro(cute))
print(cute.std(), direct.std())

# bc_direct, _ = boxcox(direct)
# bc_cute, _ = boxcox(cute)
# print(shapiro(bc_direct))
# print(shapiro(bc_cute))

# print(np.array(bc_cute).std(), np.array(bc_direct).std())
# plt.figure(figsize=(10, 7))
# n_bins = len(direct) // 2
# plt.hist(direct, bins=n_bins)
# plt.title('Receptivity cute-direct')
# plt.show()

# log_cute = np.log10(cute)
# print(shapiro(log_cute))
# bc_cute, _ = boxcox(cute)
# bc_direct, _ = boxcox(direct)
# sqrt_cute = np.sqrt(cute)
# print(shapiro(sqrt_cute))

# print(bc_cute.mean())
# print(bc_direct.mean())

# print(cute.mean())
# print(direct.mean())

from scipy.stats import levene, ttest_ind, ttest_rel, ttest_1samp
# lev = levene(bc_cute, bc_direct)
# print(lev)

# bc_cute = power_transform(cute, method='box-cox')
# print(shapiro(bc_cute))
# bc_direct = power_transform(direct, method='box-cox')
# print(shapiro(bc_direct))
# print(cute.mean())
# print(direct.mean())
# bc_direct.reshape(1, -1)
# bc_cute.reshape(1, -1)

from scipy.stats import levene, ttest_ind, ttest_rel, ttest_1samp
lev = levene(cute, direct)
print(lev)

# import seaborn
#
# seaborn.set(style='whitegrid')
# seaborn.boxplot(data=df, x='PickUp', y='Receptivity')
# plt.show()

# stat, p_value = mannwhitneyu(cute, direct, alternative="greater")
# print('Statistics=%.2f, p=%.2f' % (stat, p_value))

# print(49 / 5 + 4 / 5 + 25 / (50 ** 0.5))
print(chisquare([795, 705]))

# 10,10,10,5,10,15
# import random
#
# y = []
# for i in range(100):
#
#     x = []
#     for i in range(60):
#         x.append(random.randint(0, 1))
#
#     y.append((x.count(0) - 30)**2 / 30 + (x.count(1) - 30)**2 / 30)
# plt.hist(y)
# plt.show()

# print(contingency.expected_freq(([10, 6], [5, 15])))
print(chi2_contingency(([20, 15], [11, 12], [7, 9])))