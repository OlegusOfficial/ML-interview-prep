from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import power_transform
import pandas as pd
import numpy as np
import pyreadstat
from scipy.stats import shapiro, boxcox, pearsonr, pointbiserialr, skew, kurtosis, mannwhitneyu, f_oneway
import matplotlib
import matplotlib.pyplot as plt
from math import log
from statsmodels.stats.stattools import durbin_watson
from math import log10
import seaborn
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp

df, meta = pyreadstat.read_sav('PickUpLines.sav')

print(df.columns)


direct_spray = np.array(df[(df['PickUp'] == 2) & (df['Scent'] == 1)]['Receptivity'], dtype=object)
cute_spray = np.array(df[(df['PickUp'] == 1) & (df['Scent'] == 1)]['Receptivity'], dtype=object)
direct_nospray = np.array(df[(df['PickUp'] == 2) & (df['Scent'] == 2)]['Receptivity'], dtype=object)
cute_nospray = np.array(df[(df['PickUp'] == 1) & (df['Scent'] == 2)]['Receptivity'], dtype=object)

# log_cute_spray, _ = boxcox(cute_spray)

# print(len(direct_spray), len(cute_spray), len(direct_nospray), len(cute_nospray))
# print(len(df['Receptivity']))

# seaborn.set(style='whitegrid')
# seaborn.boxplot(y=cute_nospray)
# plt.show()
# print(shapiro(direct_spray))
# print(shapiro(cute_spray))
# print(shapiro(direct_nospray))
# print(shapiro(cute_nospray))

# plt.figure(figsize=(10, 7))
# n_bins = len(cute_spray) // 2
# plt.hist(cute_spray, bins=n_bins)
# plt.title('Receptivity cute-direct')
# plt.show()

# print(skew(cute_spray))

# from scipy.stats import levene, ttest_ind, ttest_rel, ttest_1samp
# lev = levene(direct_spray, cute_spray, direct_nospray, cute_nospray, center='mean')
# print(lev)

# print(cute_nospray.std() / direct_nospray.std())
# print(cute_nospray.std(), direct_nospray.std())

# model = ols('Receptivity ~ C(PickUp) + C(Scent) +\
# C(PickUp)*C(Scent)',
#             data=df).fit()
# result = sm.stats.anova_lm(model, type=1)

# Print the result
# print(result)

# print(f_oneway(direct_spray, cute_spray, direct_nospray, cute_nospray))

data = np.array([direct_spray, cute_spray, direct_nospray, cute_nospray], dtype=object)

# Conduct the Nemenyi post-hoc test
# print(sp.posthoc_dunn(data, p_adjust='bonferroni'))
print(sp.posthoc_dunn(data))

# print(cute_spray.shape[0])
# print(data)
print(direct_spray.mean(), cute_spray.mean(), direct_nospray.mean(), cute_nospray.mean())

