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
# target_list = catalog_access.phangs_galaxy_list

dist_list = []
for target in cc_target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
# sort target list after distance
sort = np.argsort(dist_list)
cc_target_list = np.array(cc_target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=cc_target_list)
catalog_access.load_hst_cc_list(target_list=cc_target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=cc_target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=cc_target_list, classify='ml', cluster_class='class3')



for band, band_name in zip(['F275W', 'F336W', 'F438W', 'F555W', 'F814W'], ['NUV', 'U', 'B', 'V', 'I']):


    max_abs_mag_ml_12 = np.zeros(len(cc_target_list))
    min_abs_mag_ml_12 = np.zeros(len(cc_target_list))
    mean_abs_mag_ml_12 = np.zeros(len(cc_target_list))
    median_abs_mag_ml_12 = np.zeros(len(cc_target_list))
    p16_abs_mag_ml_12 = np.zeros(len(cc_target_list))
    p84_abs_mag_ml_12 = np.zeros(len(cc_target_list))
    max_app_mag_ml_12 = np.zeros(len(cc_target_list))
    min_app_mag_ml_12 = np.zeros(len(cc_target_list))
    mean_app_mag_ml_12 = np.zeros(len(cc_target_list))
    median_app_mag_ml_12 = np.zeros(len(cc_target_list))
    p16_app_mag_ml_12 = np.zeros(len(cc_target_list))
    p84_app_mag_ml_12 = np.zeros(len(cc_target_list))


    for index, target in enumerate(cc_target_list):

        cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')

        if band == 'F438W':
            if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
                band_string = 'F438W'
            else:
                band_string = 'F435W'
        else:
            band_string = band

        mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band=band_string)

        mask_good_values = (mag_ml_12 > 10) & (mag_ml_12 < 30)
        mag_ml_12 = mag_ml_12[mask_good_values]
        abs_mag_ml_12 = mag_ml_12 - 5*np.log10(dist_list[index] * 1e6) + 5



        max_app_mag_ml_12[index] = np.nanmax(mag_ml_12)
        min_app_mag_ml_12[index] = np.nanmin(mag_ml_12)
        mean_app_mag_ml_12[index] = np.nanmean(mag_ml_12)
        median_app_mag_ml_12[index] = np.nanmedian(mag_ml_12)
        p16_app_mag_ml_12[index] = np.percentile(mag_ml_12, 16)
        p84_app_mag_ml_12[index] = np.percentile(mag_ml_12, 84)

        max_abs_mag_ml_12[index] = np.nanmax(abs_mag_ml_12)
        min_abs_mag_ml_12[index] = np.nanmin(abs_mag_ml_12)
        mean_abs_mag_ml_12[index] = np.nanmean(abs_mag_ml_12)
        median_abs_mag_ml_12[index] = np.nanmedian(abs_mag_ml_12)
        p16_abs_mag_ml_12[index] = np.percentile(abs_mag_ml_12, 16)
        p84_abs_mag_ml_12[index] = np.percentile(abs_mag_ml_12, 84)

        # mask_above_p16 = mag_ml_12 > p16_app_mag_ml_12[index]
        # median_of_p16 = np.nanmedian(mag_ml_12[mask_above_p16])
        # print(target, ' at d = ', dist_list[index], ' Mpc ', band_name, ' band median p16 mag = ', median_of_p16)

    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print('p16_app_mag_ml_12 ', p84_app_mag_ml_12)
    print('median of all p84 of the ', band_name, ' band =', np.nanmedian(p84_app_mag_ml_12))
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18, 16))
    fontsize = 23
    scatter_dot_size = 140
    scatter_star_size = 400

    err_bar_dot_size = 12
    err_bar_star_size = 21



    ax[0].scatter(dist_list, min_app_mag_ml_12, s=120, color='tab:blue', label='Min value')
    ax[0].errorbar(dist_list, median_app_mag_ml_12,
                      yerr=[median_app_mag_ml_12 - p16_app_mag_ml_12,
                            p84_app_mag_ml_12 - median_app_mag_ml_12],
                      fmt='o', ms=err_bar_dot_size, color='darkslategrey', label='Median value')
    ax[0].scatter(dist_list, max_app_mag_ml_12, s=120, color='tab:red', label='Max value')

    ax[1].scatter(dist_list, min_abs_mag_ml_12, s=120, color='tab:blue', label='Min value')
    ax[1].errorbar(dist_list, median_abs_mag_ml_12,
                      yerr=[median_abs_mag_ml_12 - p16_abs_mag_ml_12,
                            p84_abs_mag_ml_12 - median_abs_mag_ml_12],
                      fmt='o', ms=err_bar_dot_size, color='darkslategrey', label='Median value')
    ax[1].scatter(dist_list, max_abs_mag_ml_12, s=120, color='tab:red', label='Max value')


    ax[0].set_title('ML Class 1, 2', fontsize=fontsize+4)
    # ax[1].set_title('ML Class 1, 2', fontsize=fontsize+4)
    ax[0].legend(frameon=False, fontsize=fontsize - 6)
    ax[1].legend(frameon=False, fontsize=fontsize - 6)

    # ax[0, 0].set_ylim(-3.5, -17.1)
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')

    ax[0].set_ylabel('App. %s-mag' % band_name, fontsize=fontsize)
    ax[1].set_ylabel('Abs. %s-mag' % band_name, fontsize=fontsize)
    ax[1].set_xlabel('Distance [Mpc]', fontsize=fontsize)
    ax[0].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
    ax[1].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)

    # plt.show()
    # exit()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    plt.savefig('plot_output/mag_%s.png' % band_name)
    # plt.savefig('plot_output/mag_%s.pdf' % band_name)
    plt.cla()


exit()




for index in range(len(dist_list)):
    if not mask_deep[index]:
        ax[0, 0].scatter(dist_list[index], min_f555w_abs_mag_hum_12[index], s=scatter_star_size,  marker='*', color='tab:blue')
        ax[0, 0].errorbar(dist_list[index], median_f555w_abs_mag_hum_12[index],
                          yerr=np.array([[median_f555w_abs_mag_hum_12[index] - p16_f555w_abs_mag_hum_12[index]],
                                         [p84_f555w_abs_mag_hum_12[index] - median_f555w_abs_mag_hum_12[index]]]),
                          fmt='*', ms=err_bar_star_size, color='darkslategrey')
        ax[0, 0].scatter(dist_list[index], max_f555w_abs_mag_hum_12[index], s=scatter_star_size,  marker='*', color='tab:red')

    else:
        ax[0, 0].scatter(dist_list[index], min_f555w_abs_mag_hum_12[index], s=scatter_dot_size, color='tab:blue')
        ax[0, 0].errorbar(dist_list[index], median_f555w_abs_mag_hum_12[index],
                          yerr=np.array([[median_f555w_abs_mag_hum_12[index] - p16_f555w_abs_mag_hum_12[index]],
                                          [p84_f555w_abs_mag_hum_12[index] - median_f555w_abs_mag_hum_12[index]]]),
                          fmt='o', ms=err_bar_dot_size, color='darkslategrey')
        ax[0, 0].scatter(dist_list[index], max_f555w_abs_mag_hum_12[index], s=scatter_dot_size, color='tab:red')

ax[0, 0].scatter([], [], s=scatter_dot_size, color='tab:blue', label='Brightest cluster')
# ax[0, 0].errorbar([], [], yerr=[], fmt='o', ms=err_bar_dot_size, color='darkslategrey', label='Median value')
ax[0, 0].scatter([], [], s=scatter_dot_size, color='darkslategrey', label='Median value')
ax[0, 0].scatter([], [], s=scatter_dot_size, color='tab:red', label='Faintest cluster')


ax[0, 1].scatter(dist_list, min_f555w_abs_mag_ml_12, s=120, color='tab:blue', label='Min value')
ax[0, 1].errorbar(dist_list, median_f555w_abs_mag_ml_12,
                  yerr=[median_f555w_abs_mag_ml_12 - p16_f555w_abs_mag_ml_12,
                        p84_f555w_abs_mag_ml_12 - median_f555w_abs_mag_ml_12],
                  fmt='o', ms=err_bar_dot_size, color='darkslategrey', label='Median value')
ax[0, 1].scatter(dist_list, max_f555w_abs_mag_ml_12, s=120, color='tab:red', label='Max value')



ax[1, 0].scatter(dist_list, max_mstar_hum_12, s=scatter_dot_size, color='tab:purple')
ax[1, 0].errorbar(dist_list, median_mstar_hum_12,
                  yerr=[median_mstar_hum_12 - p16_mstar_hum_12,
                        p84_mstar_hum_12 - median_mstar_hum_12],
                  fmt='o', ms=err_bar_dot_size, color='darkslategrey')
ax[1, 0].scatter(dist_list, min_mstar_hum_12, s=scatter_dot_size, color='tab:green')

ax[1, 0].scatter([], [], s=scatter_dot_size, color='tab:purple', label='99% percentile')
ax[1, 0].scatter([], [], s=scatter_dot_size, color='darkslategrey', label='Median value')
ax[1, 0].scatter([], [], s=scatter_dot_size, color='tab:green', label='1% percentile')



ax[1, 1].scatter(dist_list, max_mstar_ml_12, s=scatter_dot_size, color='tab:purple')
ax[1, 1].errorbar(dist_list, median_mstar_ml_12,
                  yerr=[median_mstar_ml_12 - p16_mstar_ml_12,
                        p84_mstar_ml_12 - median_mstar_ml_12],
                  fmt='o', ms=err_bar_dot_size, color='darkslategrey')
ax[1, 1].scatter(dist_list, min_mstar_ml_12, s=scatter_dot_size, color='tab:green')


# for i, target in enumerate(cc_target_list):
#     if target in ['ngc4826', 'ngc5068', 'ic5332', 'ngc4548', 'ngc4564', 'ngc0685', 'ngc2835']:
#         ax[0, 0].text(dist_list[i], max_f555w_abs_mag_hum_12[i], target,
#                       horizontalalignment='left', verticalalignment='top', fontsize=fontsize)
# for i, target in enumerate(cc_target_list):
#     if target in ['ngc4826', 'ngc5068', 'ic5332', 'ngc4548', 'ngc4564', 'ngc0685', 'ngc2835']:
#         ax[0, 1].text(dist_list[i], max_f555w_abs_mag_ml_12[i], target,
#                    horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize)

ax[0, 0].set_title('Human Class 1, 2', fontsize=fontsize+4)
ax[0, 1].set_title('ML Class 1, 2', fontsize=fontsize+4)
ax[0, 0].legend(frameon=False, fontsize=fontsize - 6)
ax[1, 0].legend(frameon=False, fontsize=fontsize - 6)

# ax[0, 0].set_ylim(-3.5, -17.1)
ax[0, 0].invert_yaxis()
ax[1, 0].set_yscale('log')

ax[0, 0].set_ylabel('Abs. V-mag', fontsize=fontsize)
ax[1, 0].set_ylabel(r'log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
ax[1, 0].set_xlabel('Distance [Mpc]', fontsize=fontsize)
ax[1, 1].set_xlabel('Distance [Mpc]', fontsize=fontsize)
ax[0, 0].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)

# plt.show()
# exit()
plt.tight_layout()
plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.savefig('plot_output/mag_mstar.png')
plt.savefig('plot_output/mag_mstar.pdf')


exit()



print('number_hum_12 ', number_hum_12)
print('number_ml_12 ', number_ml_12)

mean_err_frac = np.mean(ssfr_err/ssfr)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(17, 13))
fontsize = 19

for i, target in enumerate(cc_target_list):
    ax[0].plot([ssfr[i], ssfr[i]], [number_ml_12[i], number_hum_12[i]], color='grey', linestyle='--', linewidth=3)
    ax[0].text(ssfr[i], number_ml_12[i], target, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0].scatter(ssfr, number_hum_12, s=120, color='darkslategrey', label='Human-classified', zorder=10)
ax[0].scatter(ssfr, number_ml_12, s=120, color='yellowgreen', label='ML-classified', zorder=10)
ax[0].errorbar(10**(-11), 500, xerr=10**(-11) * mean_err_frac, fmt='o', color='k')
ax[0].text(10**(-11), 600, 'Mean uncertainty', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

for i in range(len(ssfr)):
    ax[1].plot([ssfr[i], ssfr[i]], [number_ml_3[i], number_hum_3[i]], color='grey', linestyle='--', linewidth=3)
ax[1].scatter(ssfr, number_ml_3, s=120, color='darkslategrey', zorder=10)
ax[1].scatter(ssfr, number_hum_3, s=120, color='yellowgreen', zorder=10)


ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')


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
