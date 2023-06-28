""" bla bla bla """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import photometry_tools
from photometry_tools.analysis_tools import AnalysisTools
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.colors import Normalize, LogNorm
from astroquery.skyview import SkyView
from astroquery.simbad import Simbad
from mega_table import RadialMegaTable, TessellMegaTable


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

area_ssfr = np.zeros(len(target_list))



for index, target in enumerate(target_list):
    print(target_list[index])
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
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

    if number_hum_12[index] > number_ml_12[index]:
        print('!!!!!, ', target)

print('number_hum_12 ', number_hum_12)
print('number_ml_12 ', number_ml_12)



fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(13, 11))
fontsize = 17

for i, target in enumerate(target_list):
    ax[0].plot([area_ssfr[i], area_ssfr[i]], [number_ml_12[i], number_hum_12[i]], color='grey', linestyle='--', linewidth=3)
    # if target in ['ngc4826', 'ngc5068', 'ic5332', 'ngc4548', 'ngc4564', 'ngc0685', 'ngc2835']:
    #     ax[0].text(sssfr[i], number_ml_12[i], target, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0].scatter(area_ssfr, number_hum_12, s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[0].scatter(area_ssfr, number_ml_12, s=120, color='yellowgreen', label='ML-classified', zorder=10)
# ax[0].errorbar(10**(-11), 500, xerr=10**(-11) * mean_err_frac, fmt='o', color='k')
# ax[0].text(10**(-11), 600, 'Mean uncertainty', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

for i in range(len(area_ssfr)):
    ax[1].plot([area_ssfr[i], area_ssfr[i]], [number_ml_3[i], number_hum_3[i]], color='grey', linestyle='--', linewidth=3)
ax[1].scatter(area_ssfr, number_hum_3, s=120, color='darkslategrey', zorder=10)
ax[1].scatter(area_ssfr, number_ml_3, s=120, color='yellowgreen', zorder=10)


ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')

ax[0].text(7*10**(-12), 2800, 'Class 1, 2', horizontalalignment='left', verticalalignment='center', fontsize=fontsize+4)
ax[1].text(7*10**(-12), 4000, 'Class 3', horizontalalignment='left', verticalalignment='center', fontsize=fontsize+4)

# ax[0].set_title('Class 1 + 2', fontsize=fontsize+4)
# ax[1].set_title('Class 3', fontsize=fontsize+4)
ax[0].legend(frameon=False, fontsize=fontsize, loc=4)

ax[0].set_ylabel('# Clusters', fontsize=fontsize)
ax[1].set_ylabel('# Clusters', fontsize=fontsize)
ax[1].set_xlabel('log(SFR/M$_{*}$) [yr$^{-1}$] | HST Footprint', fontsize=fontsize)
ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


plt.tight_layout()
plt.savefig('plot_output/n_cluster_ssfr.png')
plt.savefig('plot_output/n_cluster_ssfr.pdf')


exit()




dss_cutout_size = {'ngc4826': 18, 'ngc5068': 18, 'ngc3621': 18, 'ic5332': 18,
                   'ngc6744': 18, 'ngc2903': 18, 'ngc0628': 18, 'ngc3351': 18,
                   'ngc3627': 18, 'ngc2835': 18, 'ic1954': 18, 'ngc4254': 18,
                   'ngc1097': 18, 'ngc5248': 18, 'ngc4571': 18, 'ngc4298': 18,
                   'ngc4689': 18, 'ngc4321': 18, 'ngc4569': 18, 'ngc4535': 18,
                   'ngc1087': 18, 'ngc1792': 18, 'ngc4548': 18, 'ngc4536': 18,
                   'ngc4303': 18, 'ngc1385': 18, 'ngc1566': 18, 'ngc1512': 18,
                   'ngc7496': 18, 'ngc1433': 18, 'ngc1300': 18, 'ngc1317': 18,
                   'ngc1672': 18, 'ngc1559': 18, 'ngc1365': 18, 'ngc0685': 18,
                   'ngc4654': 18, 'ngc2775': 18}
