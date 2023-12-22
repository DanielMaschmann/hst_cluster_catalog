""" bla bla bla """
import numpy as np
import matplotlib.pyplot as plt
import photometry_tools
from mega_table import TessellMegaTable


# get access to HST cluster catalog
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

target_list = catalog_access.target_hst_cc
#target_list = catalog_access.phangs_galaxy_list

catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


number_hum_12 = np.zeros(len(target_list))
number_hum_3 = np.zeros(len(target_list))
number_hum_all = np.zeros(len(target_list))
number_ml_12 = np.zeros(len(target_list))
number_ml_3 = np.zeros(len(target_list))
number_ml_all = np.zeros(len(target_list))


glob_ssfr = np.zeros(len(target_list))
glob_ssfr_err = np.zeros(len(target_list))
glob_log_ssfr = np.zeros(len(target_list))
glob_log_ssfr_err = np.zeros(len(target_list))

glob_sfr = np.zeros(len(target_list))
glob_log_sfr = np.zeros(len(target_list))



area_ssfr = np.zeros(len(target_list))

ssfr = np.zeros(len(target_list))

smallest_hum_all = 'None'
smallest_hum_12 = 'None'
smallest_ml_all = 'None'
smallest_ml_12 = 'None'

smallest_n_hum_all = 10000
smallest_n_hum_12 = 10000
smallest_n_ml_all = 10000
smallest_n_ml_12 = 10000

largest_hum_all = 'None'
largest_hum_12 = 'None'
largest_ml_all = 'None'
largest_ml_12 = 'None'

largest_n_hum_all = 0
largest_n_hum_12 = 0
largest_n_ml_all = 0
largest_n_ml_12 = 0

for index, target in enumerate(target_list):
    print(target_list[index])
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target

    ssfr[index] = catalog_access.get_target_ssfr(target=galaxy_name)

    ra, dec = catalog_access.get_hst_cc_coords_world(target=target)
    # load mega tables
    mega_table = TessellMegaTable.read('/home/benutzer/data/PHANGS_products/mega_tables/v3p0/hexagon/%s_base_hexagon_1p5kpc.ecsv' % galaxy_name.upper())

    hex_sig_sfr = mega_table['Sigma_SFR']
    hex_sig_sfr_err = mega_table['e_Sigma_SFR']
    hex_sig_star = mega_table['Sigma_star'] * 1e6
    hex_sig_star_err = mega_table['e_Sigma_star'] * 1e6

    hex_ids = mega_table['ID']
    mask_id_with_cluster = np.zeros(len(hex_ids), dtype=bool)

    for cluster_index in range(len(ra)):
        hex_id_of_cluster = mega_table.find_coords_in_regions(ra=ra[cluster_index], dec=dec[cluster_index], fill_value=-1)
        # print('hex_id_of_cluster ', hex_id_of_cluster)
        index_of_hex_id = np.where(hex_ids == hex_id_of_cluster[0])
        mask_id_with_cluster[index_of_hex_id] = True

    sig_star = np.log10(np.nanmean(hex_sig_star[mask_id_with_cluster]).value)
    sig_sfr = np.log10(np.nanmean(hex_sig_sfr[mask_id_with_cluster]).value)
    area_ssfr[index] = 10 ** (sig_sfr - sig_star)

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')

    number_hum_12[index] = sum(cluster_class_hum_12 == 1) + sum(cluster_class_hum_12 == 2)
    number_hum_3[index] = sum(cluster_class_hum_3 == 3)
    number_hum_all[index] = number_hum_12[index] + number_hum_3[index]

    number_ml_12[index] = sum(cluster_class_ml_12 == 1) + sum(cluster_class_ml_12 == 2)
    number_ml_3[index] = sum(cluster_class_ml_3 == 3)
    number_ml_all[index] = number_ml_12[index] + number_ml_3[index]

    glob_ssfr[index] = catalog_access.get_target_ssfr(target=galaxy_name)
    glob_ssfr_err[index] = catalog_access.get_target_ssfr_err(target=galaxy_name)
    glob_log_ssfr[index] = catalog_access.get_target_log_ssfr(target=galaxy_name)
    glob_log_ssfr_err[index] = catalog_access.get_target_log_ssfr_err(target=galaxy_name)

    glob_sfr[index] = catalog_access.get_target_sfr(target=galaxy_name)
    glob_log_sfr[index] = catalog_access.get_target_log_sfr(target=galaxy_name)

    if number_hum_12[index] > number_ml_12[index]:
        print('!!!!!, ', target)

    if (number_hum_12[index] + number_hum_3[index]) < smallest_n_hum_all:
        smallest_n_hum_all = number_hum_12[index] + number_hum_3[index]
        smallest_hum_all = target
    if number_hum_12[index] < smallest_n_hum_12:
        smallest_n_hum_12 = number_hum_12[index]
        smallest_hum_12 = target
    if (number_ml_12[index] + number_ml_3[index]) < smallest_n_ml_all:
        smallest_n_ml_all = number_ml_12[index] + number_ml_3[index]
        smallest_ml_all = target
    if number_ml_12[index] < smallest_n_ml_12:
        smallest_n_ml_12 = number_ml_12[index]
        smallest_ml_12 = target

    if (number_hum_12[index] + number_hum_3[index]) > largest_n_hum_all:
        largest_n_hum_all = number_hum_12[index] + number_hum_3[index]
        largest_hum_all = target
    if number_hum_12[index] > largest_n_hum_12:
        largest_n_hum_12 = number_hum_12[index]
        largest_hum_12 = target
    if (number_ml_12[index] + number_ml_3[index]) > largest_n_ml_all:
        largest_n_ml_all = number_ml_12[index] + number_ml_3[index]
        largest_ml_all = target
    if number_ml_12[index] > largest_n_ml_12:
        largest_n_ml_12 = number_ml_12[index]
        largest_ml_12 = target


print('smallest and largest numbers ')
print('smallest_n_hum_all, smallest_hum_all ', smallest_n_hum_all, smallest_hum_all)
print('smallest_n_hum_12, smallest_hum_12 ', smallest_n_hum_12, smallest_hum_12)
print('smallest_n_ml_all, smallest_ml_all ', smallest_n_ml_all, smallest_ml_all)
print('smallest_n_ml_12, smallest_ml_12 ', smallest_n_ml_12, smallest_ml_12)

print('largest_n_hum_all, largest_hum_all ', largest_n_hum_all, largest_hum_all)
print('largest_n_hum_12, largest_hum_12 ', largest_n_hum_12, largest_hum_12)
print('largest_n_ml_all, largest_ml_all ', largest_n_ml_all, largest_ml_all)
print('largest_n_ml_12, largest_ml_12 ', largest_n_ml_12, largest_ml_12)



print('correlations with pearson coefficient ')

print('area sSFR # hum 12', np.corrcoef(x=np.log10(area_ssfr), y=np.log10(number_hum_12))[0, 1])
print('area sSFR # ml 12', np.corrcoef(x=np.log10(area_ssfr), y=np.log10(number_ml_12))[0, 1])
print('glob sSFR # hum 12', np.corrcoef(x=np.log10(glob_ssfr), y=np.log10(number_hum_12))[0, 1])
print('glob sSFR # ml 12', np.corrcoef(x=np.log10(glob_ssfr), y=np.log10(number_ml_12))[0, 1])
print('glob SFR # hum 12', np.corrcoef(x=np.log10(glob_sfr), y=np.log10(number_hum_12))[0, 1])
print('glob SFR # ml 12', np.corrcoef(x=np.log10(glob_sfr), y=np.log10(number_ml_12))[0, 1])

print('area sSFR # hum 3', np.corrcoef(x=np.log10(area_ssfr), y=np.log10(number_hum_3))[0, 1])
print('area sSFR # ml 3', np.corrcoef(x=np.log10(area_ssfr), y=np.log10(number_ml_3))[0, 1])
print('glob sSFR # hum 3', np.corrcoef(x=np.log10(glob_ssfr), y=np.log10(number_hum_3))[0, 1])
print('glob sSFR # ml 3', np.corrcoef(x=np.log10(glob_ssfr), y=np.log10(number_ml_3))[0, 1])
print('glob SFR # hum 3', np.corrcoef(x=np.log10(glob_sfr), y=np.log10(number_hum_3))[0, 1])
print('glob SFR # ml 3', np.corrcoef(x=np.log10(glob_sfr), y=np.log10(number_ml_3))[0, 1])



fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(13, 10))
fontsize = 26

for i, target in enumerate(target_list):
    ax[0].plot([glob_sfr[i], glob_sfr[i]], [number_ml_12[i], number_hum_12[i]], color='grey', linestyle='--', linewidth=3)
    # if target in ['ngc4826', 'ngc5068', 'ic5332', 'ngc4548', 'ngc4564', 'ngc0685', 'ngc2835']:
    #     ax[0].text(sssfr[i], number_ml_12[i], target, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0].scatter(glob_sfr, number_hum_12, s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[0].scatter(glob_sfr, number_ml_12, s=120, color='yellowgreen', label='ML-classified', zorder=10)
# ax[0].errorbar(10**(-11), 500, xerr=10**(-11) * mean_err_frac, fmt='o', color='k')
# ax[0].text(10**(-11), 600, 'Mean uncertainty', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

for i in range(len(glob_sfr)):
    ax[1].plot([glob_sfr[i], glob_sfr[i]], [number_ml_3[i], number_hum_3[i]], color='grey', linestyle='--', linewidth=3)
ax[1].scatter(glob_sfr, number_hum_3, s=120, color='darkslategrey', zorder=10)
ax[1].scatter(glob_sfr, number_ml_3, s=120, color='yellowgreen', zorder=10)


ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')

# ax[0].text(6*10**(-12), 2700, 'Class 1, 2', horizontalalignment='left', verticalalignment='center', fontsize=fontsize+4)
# ax[1].text(6*10**(-12), 4000, 'Compact Associations', horizontalalignment='left', verticalalignment='center', fontsize=fontsize+4)
ax[0].text(0.02, 0.95, 'Class 1, 2', horizontalalignment='left', verticalalignment='top', fontsize=fontsize+4, transform=ax[0].transAxes)
ax[1].text(0.02, 0.95, 'Compact Associations', horizontalalignment='left', verticalalignment='top', fontsize=fontsize+4, transform=ax[1].transAxes)

# ax[0].set_title('Class 1 + 2', fontsize=fontsize+4)
# ax[1].set_title('Class 3', fontsize=fontsize+4)
ax[0].legend(frameon=False, fontsize=fontsize, loc=4)

ax[0].set_ylabel('# Clusters', fontsize=fontsize)
ax[1].set_ylabel('# Clusters', fontsize=fontsize)
# ax[1].set_xlabel('log(SFR/M$_{*}$) [yr$^{-1}$] | HST Footprint', fontsize=fontsize)
ax[1].set_xlabel('log(SFR) [M$_{\odot}$yr$^{-1}$]', fontsize=fontsize)
ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(left=0.06, bottom=0.07, right=0.995, top=0.995, wspace=0.01, hspace=0.01)
plt.savefig('plot_output/n_cluster_sfr.png')
plt.savefig('plot_output/n_cluster_sfr.pdf')
plt.clf()
plt.close("all")






fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(13, 10))
fontsize = 26

for i, target in enumerate(target_list):
    ax[0].plot([area_ssfr[i], area_ssfr[i]], [number_ml_12[i], number_hum_12[i]], color='grey', linestyle='--', linewidth=3)
ax[0].scatter(area_ssfr, number_hum_12, s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[0].scatter(area_ssfr, number_ml_12, s=120, color='yellowgreen', label='ML-classified', zorder=10)

for i in range(len(area_ssfr)):
    ax[1].plot([area_ssfr[i], area_ssfr[i]], [number_ml_3[i], number_hum_3[i]], color='grey', linestyle='--', linewidth=3)
ax[1].scatter(area_ssfr, number_hum_3, s=120, color='darkslategrey', zorder=10)
ax[1].scatter(area_ssfr, number_ml_3, s=120, color='yellowgreen', zorder=10)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')

# ax[0].text(6*10**(-12), 2700, 'Class 1, 2', horizontalalignment='left', verticalalignment='center', fontsize=fontsize+4)
# ax[1].text(6*10**(-12), 4000, 'Compact Associations', horizontalalignment='left', verticalalignment='center', fontsize=fontsize+4)

ax[0].text(0.02, 0.95, 'Class 1, 2', horizontalalignment='left', verticalalignment='top', fontsize=fontsize, transform=ax[0].transAxes)
ax[1].text(0.02, 0.95, 'Class 3 Compact Associations', horizontalalignment='left', verticalalignment='top', fontsize=fontsize, transform=ax[1].transAxes)


ax[0].legend(frameon=False, fontsize=fontsize-4, loc=4)

ax[0].set_ylabel('# Clusters', labelpad=-1, fontsize=fontsize)
ax[1].set_ylabel('# Clusters', labelpad=-1, fontsize=fontsize)
ax[1].set_xlabel('log(SFR/M$_{*}$) [yr$^{-1}$] | HST Footprint', labelpad=-1, fontsize=fontsize)
# ax[1].set_xlabel('log(SFR) [M$_{\odot}$yr$^{-1}$]', fontsize=fontsize)
ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(left=0.08, bottom=0.082, right=0.995, top=0.995, wspace=0.01, hspace=0.01)
plt.savefig('plot_output/n_cluster_ssfr.png')
plt.savefig('plot_output/n_cluster_ssfr.pdf')
plt.clf()
plt.close("all")

