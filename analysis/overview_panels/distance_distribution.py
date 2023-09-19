import numpy as np
import matplotlib.pyplot as plt
from photometry_tools.data_access import CatalogAccess


mean_dist_gc = np.load('data_output/mean_dist_gc.npy')
median_dist_gc = np.load('data_output/median_dist_gc.npy')
std_dist_gc = np.load('data_output/std_dist_gc.npy')
mean_dist_cascade = np.load('data_output/mean_dist_cascade.npy')
median_dist_cascade = np.load('data_output/median_dist_cascade.npy')
std_dist_cascade = np.load('data_output/std_dist_cascade.npy')
mean_dist_young = np.load('data_output/mean_dist_young.npy')
median_dist_young = np.load('data_output/median_dist_young.npy')
std_dist_young = np.load('data_output/std_dist_young.npy')
mean_dist_random = np.load('data_output/mean_dist_random.npy')
median_dist_random = np.load('data_output/median_dist_random.npy')
std_dist_random = np.load('data_output/std_dist_random.npy')

galaxy_name_list = ['ic1954', 'ic5332', 'ngc0628',
                     'ngc0685', 'ngc1087', 'ngc1097',
                     'ngc1300', 'ngc1317', 'ngc1365',
                     'ngc1385', 'ngc1433', 'ngc1512',
                     'ngc1559', 'ngc1566', 'ngc1672',
                     'ngc1792', 'ngc2775', 'ngc2835',
                     'ngc2903', 'ngc3351', 'ngc3621',
                     'ngc3627', 'ngc4254', 'ngc4298',
                     'ngc4303', 'ngc4321', 'ngc4535',
                     'ngc4536', 'ngc4548', 'ngc4569',
                     'ngc4571', 'ngc4654', 'ngc4689',
                     'ngc4826', 'ngc5068', 'ngc5248',
                     'ngc6744', 'ngc7496']

print(np.where(median_dist_gc < 120))
print(np.where(median_dist_cascade < 120))

print(galaxy_name_list[18])
print(galaxy_name_list[20])
print(galaxy_name_list[33])

sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = CatalogAccess(sample_table_path=sample_table_path)
catalog_access.load_sample_table()
list_delta_ms = np.zeros(len(galaxy_name_list))
for index, target in enumerate(galaxy_name_list):
    list_delta_ms[index] = catalog_access.get_target_delta_ms(target=target)

bins = np.linspace(0, 800, 21)

fig, ax = plt.subplots(ncols=1, nrows=1, sharex='all', figsize=(15, 9))
fontsize = 20

ax.hist(median_dist_young, bins=bins, histtype='step', color='b', linestyle='-', linewidth=4, label='Young Clusters (ML)')
ax.hist(median_dist_gc, bins=bins, histtype='step', color='r', linestyle='--', linewidth=4, label='Old Globular Clusters (ML)')
ax.hist(median_dist_cascade, bins=bins, histtype='step', color='g', linestyle=':', linewidth=4, label='Middle Aged Clusters (ML)')

# ax[1].scatter(median_dist_young, list_delta_ms)

ax.set_xlabel('Median dist to closest GMC [pc]', fontsize=fontsize)
ax.set_ylabel('# Counts', fontsize=fontsize)


ax.set_xlim(-10, 810)

ax.legend(frameon=False, fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

# plt.show()
plt.tight_layout()
plt.savefig('plot_output/median_dist.png')
plt.savefig('plot_output/median_dist.pdf')





