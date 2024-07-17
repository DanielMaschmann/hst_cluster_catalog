import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits

from scipy import stats

from astropy.table import Table





# Load FITS table

fits_filename = '/home/benutzer/data/PHANGS_products/HST_catalogs/SEDfix_NewModelsHSTHaUnionHaFLAG11pc_inclusiveGCcc_inclusiveGCclass_Jun21/agg_ml12_HSTHa.fits'

# fits_filename = 'agg_ML12_NBHa.fits'

table = Table.read(fits_filename)

#cut back to only massive clusters

table=table[table['SEDfix_mass']>=1.e4]

#table=table[table['YRO']==True]

# Assuming column1 is the x-axis and column2 is the y-axis

column1 = table['SEDfix_age']*1.e6

column2 = table['SEDfix_ebv']





# Add 20% random scatter to column1

scatter_percent = 0.20  # 20% scatter

column1_scattered = column1 * (1 + scatter_percent * np.random.uniform(-1, 1, size=len(column1)))



# Sorting based on column1_scattered

sort_indices = np.argsort(column1_scattered)

column1_sorted = column1_scattered[sort_indices]

column2_sorted = column2[sort_indices]



# Take log10 of column1

log_column1_sorted = np.log10(column1_sorted)



# Define window width in terms of separation on the x-axis

window_width = .5

print(window_width)



# Calculate interquartile range (IQR) in a sliding window manner

n_points = len(log_column1_sorted)

iqr_values = np.zeros(n_points)

values16 = np.zeros(n_points)

values50 = np.zeros(n_points)

values84 = np.zeros(n_points)



for i in range(n_points):

    start_idx = np.searchsorted(log_column1_sorted, log_column1_sorted[i] - window_width/2, side='right')

    end_idx = np.searchsorted(log_column1_sorted, log_column1_sorted[i] + window_width/2, side='left')

    window_data = column2_sorted[start_idx:end_idx]

#    print(i,log_column1_sorted[i],np.min(log_column1_sorted[start_idx:end_idx]), np.max(log_column1_sorted[start_idx:end_idx]), np.percentile(window_data,16),np.percentile(window_data,50),np.percentile(window_data,84))

    iqr_values[i] = stats.iqr(window_data)

    values16[i] = np.percentile(window_data,16)

    values50[i] = np.percentile(window_data,50)

    values84[i] = np.percentile(window_data,84)



# Plotting

plt.figure(figsize=(10, 6))

plt.scatter(log_column1_sorted, column2_sorted, s=2, label='Data Points',alpha=0.3)

plt.plot(log_column1_sorted,values50)

np.save('log_column1_sorted.npy', log_column1_sorted)
np.save('values50.npy', values50)
np.save('values84.npy', values84)


# Overlay shaded region for 16-84percentile range

plt.fill_between(log_column1_sorted, values16, values84, color='gray', alpha=0.5, label='16-to-84 percentile')



plt.xlabel('log SEDfix_age [yr]')

plt.ylabel('E(B-V) [mag]')

#plt.title('Machine Learning Class 1+2 Clusters, Mc >= 1e4, HST H-alpha tree')

plt.title('Machine Learning Class 1+2 Clusters, Mc >= 1e4, ground-based H-alpha tree')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.xlim(np.log10(0.5e6),np.log10(5.e10))

plt.ylim(-0.1,1.4)

plt.show()