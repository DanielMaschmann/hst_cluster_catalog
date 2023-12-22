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


# get access to HST cluster catalog
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

cc_target_list = catalog_access.target_hst_cc
target_list = catalog_access.phangs_galaxy_list

catalog_access.load_hst_cc_list(target_list=cc_target_list)
catalog_access.load_hst_cc_list(target_list=cc_target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=cc_target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=cc_target_list, classify='ml', cluster_class='class3')


number_hum_12 = np.zeros(len(target_list))
number_hum_3 = np.zeros(len(target_list))
number_hum_all = np.zeros(len(target_list))
number_ml_12 = np.zeros(len(target_list))
number_ml_3 = np.zeros(len(target_list))
number_ml_all = np.zeros(len(target_list))

sfr = np.zeros(len(target_list))
mstar = np.zeros(len(target_list))
ssfr = np.zeros(len(target_list))
delta_ms = np.zeros(len(target_list))


for index, target in enumerate(target_list):
    print(target)
    if target == 'ngc0628':
        cluster_class_hum_12_c = catalog_access.get_hst_cc_class_human(target='ngc0628c')
        cluster_class_hum_3_c = catalog_access.get_hst_cc_class_human(target='ngc0628c', cluster_class='class3')
        cluster_class_hum_12_e = catalog_access.get_hst_cc_class_human(target='ngc0628e')
        cluster_class_hum_3_e = catalog_access.get_hst_cc_class_human(target='ngc0628e', cluster_class='class3')
        cluster_class_hum_12 = np.concatenate([cluster_class_hum_12_c, cluster_class_hum_12_e])
        cluster_class_hum_3 = np.concatenate([cluster_class_hum_3_c, cluster_class_hum_3_e])

        cluster_class_ml_12_c = catalog_access.get_hst_cc_class_ml_vgg(target='ngc0628c', classify='ml')
        cluster_class_ml_3_c = catalog_access.get_hst_cc_class_ml_vgg(target='ngc0628c', classify='ml', cluster_class='class3')
        cluster_class_ml_12_e = catalog_access.get_hst_cc_class_ml_vgg(target='ngc0628e', classify='ml')
        cluster_class_ml_3_e = catalog_access.get_hst_cc_class_ml_vgg(target='ngc0628e', classify='ml', cluster_class='class3')
        cluster_class_ml_12 = np.concatenate([cluster_class_ml_12_c, cluster_class_ml_12_e])
        cluster_class_ml_3 = np.concatenate([cluster_class_ml_3_c, cluster_class_ml_3_e])
    else:
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

    sfr[index] = catalog_access.get_target_sfr(target=target)
    mstar[index] = catalog_access.get_target_mstar(target=target)
    ssfr[index] = catalog_access.get_target_ssfr(target=target)
    delta_ms[index] = catalog_access.get_target_delta_ms(target=target)

print('number_hum_12 ', number_hum_12)
print('number_ml_12 ', number_ml_12)


fig, ax = plt.subplots(nrows=4, ncols=2, sharex='row', sharey=True, figsize=(20, 25))
fontsize = 25

for i, target in enumerate(target_list):
    ax[0, 0].plot([np.log10(sfr[i]), np.log10(sfr[i])], [np.log10(number_ml_12[i]), np.log10(number_hum_12[i])], color='grey', linestyle='--', linewidth=3)
    ax[0, 1].plot([np.log10(sfr[i]), np.log10(sfr[i])], [np.log10(number_ml_3[i]), np.log10(number_hum_3[i])], color='grey', linestyle='--', linewidth=3)
ax[0, 0].scatter(np.log10(sfr), np.log10(number_hum_12), s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[0, 0].scatter(np.log10(sfr), np.log10(number_ml_12), s=120, color='yellowgreen', label='ML-classified', zorder=10)
ax[0, 1].scatter(np.log10(sfr), np.log10(number_hum_3), s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[0, 1].scatter(np.log10(sfr), np.log10(number_ml_3), s=120, color='yellowgreen', label='ML-classified', zorder=10)

coeff_sfr_hum_12 = np.corrcoef(x=np.log10(sfr), y=np.log10(number_hum_12))[0, 1]
coeff_sfr_ml_12 = np.corrcoef(x=np.log10(sfr), y=np.log10(number_ml_12))[0, 1]
coeff_sfr_hum_3 = np.corrcoef(x=np.log10(sfr), y=np.log10(number_hum_3))[0, 1]
coeff_sfr_ml_3 = np.corrcoef(x=np.log10(sfr), y=np.log10(number_ml_3))[0, 1]
ax[0, 0].text(0.03, 0.97, 'P(Hum) = %.2f' % coeff_sfr_hum_12,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[0, 0].transAxes)
ax[0, 0].text(0.03, 0.9, 'P(ML) = %.2f' % coeff_sfr_ml_12,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[0, 0].transAxes)
ax[0, 1].text(0.03, 0.97, 'P(Hum) = %.2f' % coeff_sfr_hum_3,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[0, 1].transAxes)
ax[0, 1].text(0.03, 0.9, 'P(ML) = %.2f' % coeff_sfr_ml_3,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[0, 1].transAxes)


for i, target in enumerate(target_list):
    ax[1, 0].plot([np.log10(mstar[i]), np.log10(mstar[i])], [np.log10(number_ml_12[i]), np.log10(number_hum_12[i])], color='grey', linestyle='--', linewidth=3)
    ax[1, 1].plot([np.log10(mstar[i]), np.log10(mstar[i])], [np.log10(number_ml_3[i]), np.log10(number_hum_3[i])], color='grey', linestyle='--', linewidth=3)
ax[1, 0].scatter(np.log10(mstar), np.log10(number_hum_12), s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[1, 0].scatter(np.log10(mstar), np.log10(number_ml_12), s=120, color='yellowgreen', label='ML-classified', zorder=10)
ax[1, 1].scatter(np.log10(mstar), np.log10(number_hum_3), s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[1, 1].scatter(np.log10(mstar), np.log10(number_ml_3), s=120, color='yellowgreen', label='ML-classified', zorder=10)

coeff_mstar_hum_12 = np.corrcoef(x=np.log10(mstar), y=np.log10(number_hum_12))[0, 1]
coeff_mstar_ml_12 = np.corrcoef(x=np.log10(mstar), y=np.log10(number_ml_12))[0, 1]
coeff_mstar_hum_3 = np.corrcoef(x=np.log10(mstar), y=np.log10(number_hum_3))[0, 1]
coeff_mstar_ml_3 = np.corrcoef(x=np.log10(mstar), y=np.log10(number_ml_3))[0, 1]
ax[1, 0].text(0.03, 0.97, 'P(Hum) = %.2f' % coeff_mstar_hum_12,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[1, 0].transAxes)
ax[1, 0].text(0.03, 0.9, 'P(ML) = %.2f' % coeff_mstar_ml_12,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[1, 0].transAxes)
ax[1, 1].text(0.03, 0.97, 'P(Hum) = %.2f' % coeff_mstar_hum_3,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[1, 1].transAxes)
ax[1, 1].text(0.03, 0.9, 'P(ML) = %.2f' % coeff_mstar_ml_3,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[1, 1].transAxes)



for i, target in enumerate(target_list):
    ax[2, 0].plot([np.log10(ssfr[i]), np.log10(ssfr[i])], [np.log10(number_ml_12[i]), np.log10(number_hum_12[i])], color='grey', linestyle='--', linewidth=3)
    ax[2, 1].plot([np.log10(ssfr[i]), np.log10(ssfr[i])], [np.log10(number_ml_3[i]), np.log10(number_hum_3[i])], color='grey', linestyle='--', linewidth=3)
ax[2, 0].scatter(np.log10(ssfr), np.log10(number_hum_12), s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[2, 0].scatter(np.log10(ssfr), np.log10(number_ml_12), s=120, color='yellowgreen', label='ML-classified', zorder=10)
ax[2, 1].scatter(np.log10(ssfr), np.log10(number_hum_3), s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[2, 1].scatter(np.log10(ssfr), np.log10(number_ml_3), s=120, color='yellowgreen', label='ML-classified', zorder=10)

coeff_ssfr_hum_12 = np.corrcoef(x=np.log10(ssfr), y=np.log10(number_hum_12))[0, 1]
coeff_ssfr_ml_12 = np.corrcoef(x=np.log10(ssfr), y=np.log10(number_ml_12))[0, 1]
coeff_ssfr_hum_3 = np.corrcoef(x=np.log10(ssfr), y=np.log10(number_hum_3))[0, 1]
coeff_ssfr_ml_3 = np.corrcoef(x=np.log10(ssfr), y=np.log10(number_ml_3))[0, 1]
ax[2, 0].text(0.03, 0.97, 'P(Hum) = %.2f' % coeff_ssfr_hum_12,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[2, 0].transAxes)
ax[2, 0].text(0.03, 0.9, 'P(ML) = %.2f' % coeff_ssfr_ml_12,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[2, 0].transAxes)
ax[2, 1].text(0.03, 0.97, 'P(Hum) = %.2f' % coeff_ssfr_hum_3,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[2, 1].transAxes)
ax[2, 1].text(0.03, 0.9, 'P(ML) = %.2f' % coeff_ssfr_ml_3,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[2, 1].transAxes)



for i, target in enumerate(target_list):
    ax[3, 0].plot([delta_ms[i], delta_ms[i]], [np.log10(number_ml_12[i]), np.log10(number_hum_12[i])], color='grey', linestyle='--', linewidth=3)
    ax[3, 1].plot([delta_ms[i], delta_ms[i]], [np.log10(number_ml_3[i]), np.log10(number_hum_3[i])], color='grey', linestyle='--', linewidth=3)
ax[3, 0].scatter(delta_ms, np.log10(number_hum_12), s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[3, 0].scatter(delta_ms, np.log10(number_ml_12), s=120, color='yellowgreen', label='ML-classified', zorder=10)
ax[3, 1].scatter(delta_ms, np.log10(number_hum_3), s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[3, 1].scatter(delta_ms, np.log10(number_ml_3), s=120, color='yellowgreen', label='ML-classified', zorder=10)

coeff_delta_ms_hum_12 = np.corrcoef(x=delta_ms, y=np.log10(number_hum_12))[0, 1]
coeff_delta_ms_ml_12 = np.corrcoef(x=delta_ms, y=np.log10(number_ml_12))[0, 1]
coeff_delta_ms_hum_3 = np.corrcoef(x=delta_ms, y=np.log10(number_hum_3))[0, 1]
coeff_delta_ms_ml_3 = np.corrcoef(x=delta_ms, y=np.log10(number_ml_3))[0, 1]
ax[3, 0].text(0.03, 0.97, 'P(Hum) = %.2f' % coeff_delta_ms_hum_12,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[3, 0].transAxes)
ax[3, 0].text(0.03, 0.9, 'P(ML) = %.2f' % coeff_delta_ms_ml_12,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[3, 0].transAxes)
ax[3, 1].text(0.03, 0.97, 'P(Hum) = %.2f' % coeff_delta_ms_hum_3,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[3, 1].transAxes)
ax[3, 1].text(0.03, 0.9, 'P(ML) = %.2f' % coeff_delta_ms_ml_3,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize-2, transform=ax[3, 1].transAxes)




# ax[0, 0].set_xscale('log')
# ax[0, 0].set_yscale('log')

# ax[0].text(7*10**(-12), 2800, 'Class 1, 2', horizontalalignment='left', verticalalignment='center', fontsize=fontsize+4)
# ax[1].text(7*10**(-12), 4000, 'Class 3', horizontalalignment='left', verticalalignment='center', fontsize=fontsize+4)

ax[0, 0].set_title('Class 1+2 Clusters', fontsize=fontsize+4)
ax[0, 1].set_title('Class 3 Compact Associations', fontsize=fontsize+4)
ax[0, 0].legend(frameon=False, fontsize=fontsize, loc=4)

ax[0, 0].set_ylabel('# Clusters', fontsize=fontsize)
ax[1, 0].set_ylabel('# Clusters', fontsize=fontsize)
ax[2, 0].set_ylabel('# Clusters', fontsize=fontsize)
ax[3, 0].set_ylabel('# Clusters', fontsize=fontsize)

ax[0, 0].set_xlabel('log(SFR/M$_{\odot}$ yr$^{-1}$)', fontsize=fontsize)
ax[0, 1].set_xlabel('log(SFR/M$_{\odot}$ yr$^{-1}$)', fontsize=fontsize)
ax[1, 0].set_xlabel('log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
ax[1, 1].set_xlabel('log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
ax[2, 0].set_xlabel('log(sSFR/yr$^{-1}$)', fontsize=fontsize)
ax[2, 1].set_xlabel('log(sSFR/yr$^{-1}$)', fontsize=fontsize)
ax[3, 0].set_xlabel('log($\Delta$MS/M$_{\odot}$ yr$^{-1}$)', fontsize=fontsize)
ax[3, 1].set_xlabel('log($\Delta$MS/M$_{\odot}$ yr$^{-1}$)', fontsize=fontsize)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[3, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[3, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


plt.tight_layout()
plt.savefig('plot_output/n_cluster_panel.png')
plt.savefig('plot_output/n_cluster_panel.pdf')


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
