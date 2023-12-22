import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from scipy.stats import gaussian_kde
from scipy import stats

from xgaltool import analysis_tools, plotting_tools
import matplotlib

from photometry_tools.plotting_tools import DensityContours
from matplotlib import cm
from matplotlib.patches import ConnectionPatch
from scipy.spatial import ConvexHull


vi_hull_ogc_ubvi_hum_1 = np.load('../segmentation/data_output/vi_hull_ogc_ubvi_hum_1.npy')
ub_hull_ogc_ubvi_hum_1 = np.load('../segmentation/data_output/ub_hull_ogc_ubvi_hum_1.npy')
vi_hull_ogc_ubvi_ml_1 = np.load('../segmentation/data_output/vi_hull_ogc_ubvi_ml_1.npy')
ub_hull_ogc_ubvi_ml_1 = np.load('../segmentation/data_output/ub_hull_ogc_ubvi_ml_1.npy')

vi_hull_mid_ubvi_hum_1 = np.load('../segmentation/data_output/vi_hull_mid_ubvi_hum_1.npy')
ub_hull_mid_ubvi_hum_1 = np.load('../segmentation/data_output/ub_hull_mid_ubvi_hum_1.npy')
vi_hull_mid_ubvi_ml_1 = np.load('../segmentation/data_output/vi_hull_mid_ubvi_ml_1.npy')
ub_hull_mid_ubvi_ml_1 = np.load('../segmentation/data_output/ub_hull_mid_ubvi_ml_1.npy')

vi_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/vi_hull_young_ubvi_hum_3.npy')
ub_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/ub_hull_young_ubvi_hum_3.npy')
vi_hull_young_ubvi_ml_3 = np.load('../segmentation/data_output/vi_hull_young_ubvi_ml_3.npy')
ub_hull_young_ubvi_ml_3 = np.load('../segmentation/data_output/ub_hull_young_ubvi_ml_3.npy')


hull_gc_hum = ConvexHull(np.array([vi_hull_ogc_ubvi_hum_1, ub_hull_ogc_ubvi_hum_1]).T)
hull_cascade_hum = ConvexHull(np.array([vi_hull_mid_ubvi_hum_1, ub_hull_mid_ubvi_hum_1]).T)
hull_young_hum = ConvexHull(np.array([vi_hull_young_ubvi_hum_3, ub_hull_young_ubvi_hum_3]).T)

hull_gc_ml = ConvexHull(np.array([vi_hull_ogc_ubvi_ml_1, ub_hull_ogc_ubvi_ml_1]).T)
hull_cascade_ml = ConvexHull(np.array([vi_hull_mid_ubvi_ml_1, ub_hull_mid_ubvi_ml_1]).T)
hull_young_ml = ConvexHull(np.array([vi_hull_young_ubvi_ml_3, ub_hull_young_ubvi_ml_3]).T)

model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')


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
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])

catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc)
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml', cluster_class='class3')


delta_ms_list = []
sfr_list = []
mass_list = []

frac_gc_list_hum = []
frac_cascade_list_hum = []
frac_young_list_hum = []
frac_no_class_list_hum = []
frac_gc_list_ml = []
frac_cascade_list_ml = []
frac_young_list_ml = []
frac_no_class_list_ml = []

n_gc_list_hum = []
n_cascade_list_hum = []
n_young_list_hum = []
n_no_class_list_hum = []
n_gc_list_ml = []
n_cascade_list_ml = []
n_young_list_ml = []
n_no_class_list_ml = []

brightest_gc_list_hum = []
brightest_cascade_list_hum = []
brightest_young_list_hum = []
brightest_no_class_list_hum = []
brightest_gc_list_ml = []
brightest_cascade_list_ml = []
brightest_young_list_ml = []
brightest_no_class_list_ml = []


# for index in range(2):
for index in range(len(target_list)):
    target = target_list[index]

    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target

    dist = catalog_access.dist_dict[galaxy_name]['dist']

    if (target_list[index][0:3] == 'ngc') & (target_list[index][3] == '0'):
        target_name_str = target_list[index][0:3] + ' ' + target_list[index][4:]
    elif target_list[index][0:2] == 'ic':
        target_name_str = target_list[index][0:2] + ' ' + target_list[index][2:]
    elif target_list[index][0:3] == 'ngc':
        target_name_str = target_list[index][0:3] + ' ' + target_list[index][3:]
    else:
        target_name_str = target_list[index]
    target_name_str = target_name_str.upper()

    print('target ', target_name_str)

    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'
    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    mass_hum_12 = catalog_access.get_hst_cc_stellar_m(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    detect_ub_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, band='F336W') > 0))
    detect_vi_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0))
    v_mag_hum_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    abs_v_mag_hum_12 = hf.conv_mag2abs_mag(mag=v_mag_hum_12, dist=dist)

    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    mass_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    detect_ub_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0))
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))
    v_mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band='F555W')
    abs_v_mag_ml_12 = hf.conv_mag2abs_mag(mag=v_mag_ml_12, dist=dist)


    in_hull_gc_hum = hf.points_in_hull(np.array([color_vi_hum_12, color_ub_hum_12]).T, hull_gc_hum)
    in_hull_cascade_hum = hf.points_in_hull(np.array([color_vi_hum_12, color_ub_hum_12]).T, hull_cascade_hum)
    in_hull_young_hum = hf.points_in_hull(np.array([color_vi_hum_12, color_ub_hum_12]).T, hull_young_hum)

    in_hull_gc_ml = hf.points_in_hull(np.array([color_vi_ml_12, color_ub_ml_12]).T, hull_gc_ml)
    in_hull_cascade_ml = hf.points_in_hull(np.array([color_vi_ml_12, color_ub_ml_12]).T, hull_cascade_ml)
    in_hull_young_ml = hf.points_in_hull(np.array([color_vi_ml_12, color_ub_ml_12]).T, hull_young_ml)

    detectable_hum = detect_ub_hum_12 * detect_vi_hum_12 #* (mass_hum_12 > 1e4)
    detectable_ml = detect_ub_ml_12 * detect_vi_ml_12 #* (mass_ml_12 > 1e4)

    young_hum = age_hum_12 < 10
    young_ml = age_ml_12 < 10

    mask_gc_hum = in_hull_gc_hum * detectable_hum * np.invert(young_hum)
    mask_cascade_hum = in_hull_cascade_hum * detectable_hum * np.invert(young_hum)
    mask_young_hum = ((detectable_hum * young_hum * (in_hull_gc_hum + in_hull_cascade_hum)) +
                      (in_hull_young_hum * detectable_hum * np.invert(in_hull_cascade_hum)))
    mask_no_class_hum = (np.invert(in_hull_gc_hum + in_hull_cascade_hum + in_hull_young_hum) * detectable_hum)

    mask_gc_ml = in_hull_gc_ml * detectable_ml * np.invert(young_ml)
    mask_cascade_ml = in_hull_cascade_ml * detectable_ml * np.invert(young_ml)
    mask_young_ml = ((detectable_ml * young_ml * (in_hull_gc_ml + in_hull_cascade_ml)) +
                      (in_hull_young_ml * detectable_ml * np.invert(in_hull_cascade_ml)))
    mask_no_class_ml = (np.invert(in_hull_gc_ml + in_hull_cascade_ml + in_hull_young_ml) * detectable_ml)

    print("%.1f " % (sum(mask_gc_hum) / sum(detectable_hum) * 100),
          "%.1f " % (sum(mask_cascade_hum) / sum(detectable_hum) * 100),
          "%.1f " % (sum(mask_young_hum) / sum(detectable_hum) * 100),
          "%.1f " % (sum(mask_no_class_hum) / sum(detectable_hum) * 100))
    print("%.1f " % (sum(mask_gc_ml) / sum(detectable_ml) * 100),
          "%.1f " % (sum(mask_cascade_ml) / sum(detectable_ml) * 100),
          "%.1f " % (sum(mask_young_ml) / sum(detectable_ml) * 100),
          "%.1f " % (sum(mask_no_class_ml) / sum(detectable_ml) * 100))


    # x_lim_vi = (-0.7, 2.4)
    # y_lim_ub = (2.1, -2.2)
    # plt.plot(model_vi_sol, model_ub_sol, color='r')
    # plt.scatter(color_vi_ml_12[mask_gc_ml], color_ub_ml_12[mask_gc_ml], s=100, color='r')
    # plt.scatter(color_vi_ml_12[mask_cascade_ml], color_ub_ml_12[mask_cascade_ml], s=66, color='g')
    # plt.scatter(color_vi_ml_12[mask_young_ml], color_ub_ml_12[mask_young_ml], s=33, color='b')
    # plt.scatter(color_vi_ml_12[mask_no_class_ml], color_ub_ml_12[mask_no_class_ml], s=10, color='gray')
    #
    # plt.xlim(x_lim_vi)
    # plt.ylim(y_lim_ub)
    # plt.show()
    # # exit()

    if (target == 'ngc0628e') | (target == 'ngc0628c'):
        target_str = 'ngc0628'
    else:
        target_str = target

    sfr = catalog_access.get_target_sfr(target=target_str)
    sfr_err = catalog_access.get_target_sfr_err(target=target_str)
    mstar = catalog_access.get_target_mstar(target=target_str)
    mstar_err = catalog_access.get_target_mstar_err(target=target_str)
    delta_ms = catalog_access.get_target_delta_ms(target=target_str)

    delta_ms_list.append(delta_ms)
    sfr_list.append(sfr)
    mass_list.append(mstar)
    frac_gc_list_hum.append(sum(mask_gc_hum) / sum(detectable_hum))
    frac_cascade_list_hum.append(sum(mask_cascade_hum) / sum(detectable_hum))
    frac_young_list_hum.append(sum(mask_young_hum) / sum(detectable_hum))
    frac_no_class_list_hum.append(sum(mask_no_class_hum) / sum(detectable_hum))
    frac_gc_list_ml.append(sum(mask_gc_ml) / sum(detectable_ml))
    frac_cascade_list_ml.append(sum(mask_cascade_ml) / sum(detectable_ml))
    frac_young_list_ml.append(sum(mask_young_ml) / sum(detectable_ml))
    frac_no_class_list_ml.append(sum(mask_no_class_ml) / sum(detectable_ml))

    mask_bright_hum = abs_v_mag_hum_12 < -7.5
    mask_bright_ml = abs_v_mag_ml_12 < -7.5

    n_gc_list_hum.append(sum(mask_gc_hum * mask_bright_hum))
    n_cascade_list_hum.append(sum(mask_cascade_hum * mask_bright_hum))
    n_young_list_hum.append(sum(mask_young_hum * mask_bright_hum))
    n_no_class_list_hum.append(sum(mask_no_class_hum * mask_bright_hum))
    n_gc_list_ml.append(sum(mask_gc_ml * mask_bright_ml))
    n_cascade_list_ml.append(sum(mask_cascade_ml * mask_bright_ml))
    n_young_list_ml.append(sum(mask_young_ml * mask_bright_ml))
    n_no_class_list_ml.append(sum(mask_no_class_ml * mask_bright_ml))

    brightest_gc_list_hum.append(np.min(abs_v_mag_hum_12[mask_gc_hum]))
    brightest_cascade_list_hum.append(np.min(abs_v_mag_hum_12[mask_cascade_hum]))
    brightest_young_list_hum.append(np.min(abs_v_mag_hum_12[mask_young_hum]))
    brightest_no_class_list_hum.append(np.min(abs_v_mag_hum_12[mask_no_class_hum]))
    brightest_gc_list_ml.append(np.min(abs_v_mag_ml_12[mask_gc_ml]))
    brightest_cascade_list_ml.append(np.min(abs_v_mag_ml_12[mask_cascade_ml]))
    brightest_young_list_ml.append(np.min(abs_v_mag_ml_12[mask_young_ml]))
    brightest_no_class_list_ml.append(np.min(abs_v_mag_ml_12[mask_no_class_ml]))



# get pearson coeeficients
pearson_n_gc_hum = stats.pearsonr(delta_ms_list, n_gc_list_hum).statistic
pearson_n_cascade_hum = stats.pearsonr(delta_ms_list, n_cascade_list_hum).statistic
pearson_n_young_hum = stats.pearsonr(delta_ms_list, n_young_list_hum).statistic
pearson_n_no_class_hum = stats.pearsonr(delta_ms_list, n_no_class_list_hum).statistic

pearson_n_gc_ml = stats.pearsonr(delta_ms_list, n_gc_list_ml).statistic
pearson_n_cascade_ml = stats.pearsonr(delta_ms_list, n_cascade_list_ml).statistic
pearson_n_young_ml = stats.pearsonr(delta_ms_list, n_young_list_ml).statistic
pearson_n_no_class_ml = stats.pearsonr(delta_ms_list, n_no_class_list_ml).statistic

# get pearson coeeficients
pearson_brightest_gc_hum = stats.pearsonr(delta_ms_list, brightest_gc_list_hum).statistic
pearson_brightest_cascade_hum = stats.pearsonr(delta_ms_list, brightest_cascade_list_hum).statistic
pearson_brightest_young_hum = stats.pearsonr(delta_ms_list, brightest_young_list_hum).statistic
pearson_brightest_no_class_hum = stats.pearsonr(delta_ms_list, brightest_no_class_list_hum).statistic

pearson_brightest_gc_ml = stats.pearsonr(delta_ms_list, brightest_gc_list_ml).statistic
pearson_brightest_cascade_ml = stats.pearsonr(delta_ms_list, brightest_cascade_list_ml).statistic
pearson_brightest_young_ml = stats.pearsonr(delta_ms_list, brightest_young_list_ml).statistic
pearson_brightest_no_class_ml = stats.pearsonr(delta_ms_list, brightest_no_class_list_ml).statistic

mask_close = np.array(dist_list) < 15

delta_ms_list = np.array(delta_ms_list)
sfr_list = np.array(sfr_list)
mass_list = np.array(mass_list)

frac_gc_list_hum = np.array(frac_gc_list_hum)
frac_cascade_list_hum = np.array(frac_cascade_list_hum)
frac_young_list_hum = np.array(frac_young_list_hum)
frac_no_class_list_hum = np.array(frac_no_class_list_hum)

frac_gc_list_ml = np.array(frac_gc_list_ml)
frac_cascade_list_ml = np.array(frac_cascade_list_ml)
frac_young_list_ml = np.array(frac_young_list_ml)
frac_no_class_list_ml = np.array(frac_no_class_list_ml)


n_gc_list_hum = np.array(n_gc_list_hum)
n_cascade_list_hum = np.array(n_cascade_list_hum)
n_young_list_hum = np.array(n_young_list_hum)
n_no_class_list_hum = np.array(n_no_class_list_hum)

n_gc_list_ml = np.array(n_gc_list_ml)
n_cascade_list_ml = np.array(n_cascade_list_ml)
n_young_list_ml = np.array(n_young_list_ml)
n_no_class_list_ml = np.array(n_no_class_list_ml)


brightest_gc_list_hum = np.array(brightest_gc_list_hum)
brightest_cascade_list_hum = np.array(brightest_cascade_list_hum)
brightest_young_list_hum = np.array(brightest_young_list_hum)
brightest_no_class_list_hum = np.array(brightest_no_class_list_hum)

brightest_gc_list_ml = np.array(brightest_gc_list_ml)
brightest_cascade_list_ml = np.array(brightest_cascade_list_ml)
brightest_young_list_ml = np.array(brightest_young_list_ml)
brightest_no_class_list_ml = np.array(brightest_no_class_list_ml)




fig, ax = plt.subplots(ncols=4, nrows=2, sharex='all', sharey='row', figsize=(30, 17))
fontsize = 30

ax[0, 0].scatter(np.log10(sfr_list)[mask_close], n_young_list_hum[mask_close], s=200, color='tab:blue')
ax[0, 1].scatter(np.log10(sfr_list)[mask_close], n_cascade_list_hum[mask_close], s=200, color='tab:green')
ax[0, 2].scatter(np.log10(sfr_list)[mask_close], n_gc_list_hum[mask_close], s=200, color='tab:red')
ax[0, 3].scatter(np.log10(sfr_list)[mask_close], n_no_class_list_hum[mask_close], s=200, color='tab:gray')

ax[0, 0].scatter(np.log10(sfr_list)[np.invert(mask_close)], n_young_list_hum[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:blue')
ax[0, 1].scatter(np.log10(sfr_list)[np.invert(mask_close)], n_cascade_list_hum[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:green')
ax[0, 2].scatter(np.log10(sfr_list)[np.invert(mask_close)], n_gc_list_hum[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:red')
ax[0, 3].scatter(np.log10(sfr_list)[np.invert(mask_close)], n_no_class_list_hum[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:gray')



ax[1, 0].scatter(np.log10(sfr_list)[mask_close], n_young_list_ml[mask_close], s=200, color='tab:blue')
ax[1, 1].scatter(np.log10(sfr_list)[mask_close], n_cascade_list_ml[mask_close], s=200, color='tab:green')
ax[1, 2].scatter(np.log10(sfr_list)[mask_close], n_gc_list_ml[mask_close], s=200, color='tab:red')
ax[1, 3].scatter(np.log10(sfr_list)[mask_close], n_no_class_list_ml[mask_close], s=200, color='tab:gray')

ax[1, 0].scatter(np.log10(sfr_list)[np.invert(mask_close)], n_young_list_ml[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:blue')
ax[1, 1].scatter(np.log10(sfr_list)[np.invert(mask_close)], n_cascade_list_ml[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:green')
ax[1, 2].scatter(np.log10(sfr_list)[np.invert(mask_close)], n_gc_list_ml[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:red')
ax[1, 3].scatter(np.log10(sfr_list)[np.invert(mask_close)], n_no_class_list_ml[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:gray')

# # linear fit
# dummy_x = np.linspace(np.min(np.log10(sfr_list)), np.max(np.log10(sfr_list)), 10)
#
# coef_cascade_hum, cov_cascade_hum = np.polyfit(np.log10(sfr_list), frac_cascade_list_hum,1, cov=True)
# coef_cascade_ml, cov_cascade_ml = np.polyfit(np.log10(sfr_list), frac_cascade_list_ml,1, cov=True)
# poly1d_fn_cascade_hum = np.poly1d(coef_cascade_hum)
# poly1d_fn_cascade_ml = np.poly1d(coef_cascade_ml)
# uncert_cascade_hum = np.sqrt(np.diag(cov_cascade_hum))
# uncert_cascade_ml = np.sqrt(np.diag(cov_cascade_ml))
#
# print('coef_cascade_hum ', coef_cascade_hum)
# print('coef_cascade_ml ', coef_cascade_ml)
# print('uncert_cascade_hum ', uncert_cascade_hum)
# print('uncert_cascade_ml ', uncert_cascade_ml)
#
#
# ax[0, 1].plot(dummy_x, poly1d_fn_cascade_hum(dummy_x), color='k', linestyle='--', linewidth=3,
#               label=r'%.2f$(\pm %.2f)\times \Delta$MS + %.2f($\pm$%.2f)' %
#                     (coef_cascade_hum[0], uncert_cascade_hum[0], coef_cascade_hum[1], uncert_cascade_hum[1]))
# ax[1, 1].plot(dummy_x, poly1d_fn_cascade_ml(dummy_x), color='k', linestyle='--', linewidth=3,
#               label=r'%.2f$(\pm %.2f)\times \Delta$MS + %.2f($\pm$%.2f)' %
#                     (coef_cascade_ml[0], uncert_cascade_ml[0], coef_cascade_ml[1], uncert_cascade_ml[1]))
# ax[0, 1].legend(frameon=True, loc=4, fontsize=fontsize-6)
# ax[1, 1].legend(frameon=True, loc=4, fontsize=fontsize-6)

ax[0, 0].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_n_young_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 0].transAxes)
ax[0, 1].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_n_cascade_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 1].transAxes)
ax[0, 2].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_n_gc_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 2].transAxes)
ax[0, 3].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_n_no_class_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 3].transAxes)

ax[1, 0].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_n_young_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 0].transAxes)
ax[1, 1].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_n_cascade_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 1].transAxes)
ax[1, 2].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_n_gc_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 2].transAxes)
ax[1, 3].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_n_no_class_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 3].transAxes)


ax[0, 0].set_ylabel(r'N Clusters (M$_{\rm V}$ < -7.5 mag)', fontsize=fontsize)
ax[1, 0].set_ylabel(r'N Clusters (M$_{\rm V}$ < -7.5 mag)', fontsize=fontsize)

ax[1, 0].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)
ax[1, 1].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)
ax[1, 2].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)
ax[1, 3].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)

ax[0, 0].set_title('Young Cluster Locus', fontsize=fontsize)
ax[0, 1].set_title('Middle-Age Plume ', fontsize=fontsize)
ax[0, 2].set_title('Old Globular Clusters Clump', fontsize=fontsize)
ax[0, 3].set_title('Outside Main Regions', fontsize=fontsize)

ax[0, 0].text(0.03, 0.97, 'Hum', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 0].transAxes)
ax[0, 1].text(0.03, 0.97, 'Hum', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 1].transAxes)
ax[0, 2].text(0.03, 0.97, 'Hum', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 2].transAxes)
ax[0, 3].text(0.03, 0.97, 'Hum', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 3].transAxes)

ax[1, 0].text(0.03, 0.97, 'ML', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[1, 0].transAxes)
ax[1, 1].text(0.03, 0.97, 'ML', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[1, 1].transAxes)
ax[1, 2].text(0.03, 0.97, 'ML', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[1, 2].transAxes)
ax[1, 3].text(0.03, 0.97, 'ML', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[1, 3].transAxes)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

# ax[0, 0].set_xlim(-0.8, 0.8)

fig.subplots_adjust(left=0.05, bottom=0.06, right=0.995, top=0.97, wspace=0.01, hspace=0.01)
# plt.tight_layout()
fig.savefig('plot_output/ms_number_stats.png')
fig.savefig('plot_output/ms_number_stats.pdf')

plt.clf()
plt.cla()




fig, ax = plt.subplots(ncols=4, nrows=2, sharex='all', sharey='row', figsize=(30, 17))
fontsize = 30

ax[0, 0].scatter(np.log10(sfr_list)[mask_close], brightest_young_list_hum[mask_close], s=200, color='tab:blue')
ax[0, 1].scatter(np.log10(sfr_list)[mask_close], brightest_cascade_list_hum[mask_close], s=200, color='tab:green')
ax[0, 2].scatter(np.log10(sfr_list)[mask_close], brightest_gc_list_hum[mask_close], s=200, color='tab:red')
ax[0, 3].scatter(np.log10(sfr_list)[mask_close], brightest_no_class_list_hum[mask_close], s=200, color='tab:gray')

ax[0, 0].scatter(np.log10(sfr_list)[np.invert(mask_close)], brightest_young_list_hum[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:blue')
ax[0, 1].scatter(np.log10(sfr_list)[np.invert(mask_close)], brightest_cascade_list_hum[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:green')
ax[0, 2].scatter(np.log10(sfr_list)[np.invert(mask_close)], brightest_gc_list_hum[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:red')
ax[0, 3].scatter(np.log10(sfr_list)[np.invert(mask_close)], brightest_no_class_list_hum[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:gray')



ax[1, 0].scatter(np.log10(sfr_list)[mask_close], brightest_young_list_ml[mask_close], s=200, color='tab:blue')
ax[1, 1].scatter(np.log10(sfr_list)[mask_close], brightest_cascade_list_ml[mask_close], s=200, color='tab:green')
ax[1, 2].scatter(np.log10(sfr_list)[mask_close], brightest_gc_list_ml[mask_close], s=200, color='tab:red')
ax[1, 3].scatter(np.log10(sfr_list)[mask_close], brightest_no_class_list_ml[mask_close], s=200, color='tab:gray')

ax[1, 0].scatter(np.log10(sfr_list)[np.invert(mask_close)], brightest_young_list_ml[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:blue')
ax[1, 1].scatter(np.log10(sfr_list)[np.invert(mask_close)], brightest_cascade_list_ml[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:green')
ax[1, 2].scatter(np.log10(sfr_list)[np.invert(mask_close)], brightest_gc_list_ml[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:red')
ax[1, 3].scatter(np.log10(sfr_list)[np.invert(mask_close)], brightest_no_class_list_ml[np.invert(mask_close)], s=200, linewidth=4, facecolor='None', color='tab:gray')

# # linear fit
# dummy_x = np.linspace(np.min(delta_ms_list), np.max(delta_ms_list), 10)
#
# coef_cascade_hum, cov_cascade_hum = np.polyfit(delta_ms_list, frac_cascade_list_hum,1, cov=True)
# coef_cascade_ml, cov_cascade_ml = np.polyfit(delta_ms_list, frac_cascade_list_ml,1, cov=True)
# poly1d_fn_cascade_hum = np.poly1d(coef_cascade_hum)
# poly1d_fn_cascade_ml = np.poly1d(coef_cascade_ml)
# uncert_cascade_hum = np.sqrt(np.diag(cov_cascade_hum))
# uncert_cascade_ml = np.sqrt(np.diag(cov_cascade_ml))
#
# print('coef_cascade_hum ', coef_cascade_hum)
# print('coef_cascade_ml ', coef_cascade_ml)
# print('uncert_cascade_hum ', uncert_cascade_hum)
# print('uncert_cascade_ml ', uncert_cascade_ml)
#
#
# ax[0, 1].plot(dummy_x, poly1d_fn_cascade_hum(dummy_x), color='k', linestyle='--', linewidth=3,
#               label=r'%.2f$(\pm %.2f)\times \Delta$MS + %.2f($\pm$%.2f)' %
#                     (coef_cascade_hum[0], uncert_cascade_hum[0], coef_cascade_hum[1], uncert_cascade_hum[1]))
# ax[1, 1].plot(dummy_x, poly1d_fn_cascade_ml(dummy_x), color='k', linestyle='--', linewidth=3,
#               label=r'%.2f$(\pm %.2f)\times \Delta$MS + %.2f($\pm$%.2f)' %
#                     (coef_cascade_ml[0], uncert_cascade_ml[0], coef_cascade_ml[1], uncert_cascade_ml[1]))
# ax[0, 1].legend(frameon=True, loc=4, fontsize=fontsize-6)
# ax[1, 1].legend(frameon=True, loc=4, fontsize=fontsize-6)

ax[0, 0].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_brightest_young_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 0].transAxes)
ax[0, 1].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_brightest_cascade_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 1].transAxes)
ax[0, 2].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_brightest_gc_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 2].transAxes)
ax[0, 3].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_brightest_no_class_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 3].transAxes)

ax[1, 0].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_brightest_young_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 0].transAxes)
ax[1, 1].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_brightest_cascade_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 1].transAxes)
ax[1, 2].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_brightest_gc_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 2].transAxes)
ax[1, 3].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_brightest_no_class_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 3].transAxes)


ax[0, 0].set_ylabel(r' Max(M$_{\rm V}$)', fontsize=fontsize)
ax[1, 0].set_ylabel(r' Max(M$_{\rm V}$)', fontsize=fontsize)

ax[1, 0].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)
ax[1, 1].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)
ax[1, 2].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)
ax[1, 3].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)

ax[0, 0].set_title('Young Cluster Locus', fontsize=fontsize)
ax[0, 1].set_title('Middle-Age Plume ', fontsize=fontsize)
ax[0, 2].set_title('Old Globular Clusters Clump', fontsize=fontsize)
ax[0, 3].set_title('Outside Main Regions', fontsize=fontsize)

ax[0, 0].text(0.03, 0.97, 'Hum', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 0].transAxes)
ax[0, 1].text(0.03, 0.97, 'Hum', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 1].transAxes)
ax[0, 2].text(0.03, 0.97, 'Hum', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 2].transAxes)
ax[0, 3].text(0.03, 0.97, 'Hum', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 3].transAxes)

ax[1, 0].text(0.03, 0.97, 'ML', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[1, 0].transAxes)
ax[1, 1].text(0.03, 0.97, 'ML', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[1, 1].transAxes)
ax[1, 2].text(0.03, 0.97, 'ML', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[1, 2].transAxes)
ax[1, 3].text(0.03, 0.97, 'ML', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[1, 3].transAxes)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

# ax[0, 0].set_xlim(-0.8, 0.8)

fig.subplots_adjust(left=0.05, bottom=0.06, right=0.995, top=0.97, wspace=0.01, hspace=0.01)
# plt.tight_layout()
fig.savefig('plot_output/ms_brightest_stats.png')
fig.savefig('plot_output/ms_brightest_stats.pdf')
