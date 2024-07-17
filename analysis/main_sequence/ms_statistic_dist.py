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
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc)
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml', cluster_class='class3')


delta_ms_list = []
dist_list = []
dist_err_list = []
dist_method_list = []
frac_gc_list_hum = []
frac_cascade_list_hum = []
frac_young_list_hum = []
frac_no_class_list_hum = []
frac_gc_list_ml = []
frac_cascade_list_ml = []
frac_young_list_ml = []
frac_no_class_list_ml = []


# for index in range(2):
for index in range(len(target_list)):
    target = target_list[index]

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

    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    mass_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    detect_ub_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0))
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))


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
    dist = catalog_access.get_target_dist(target=target_str)
    dist_err = catalog_access.get_target_dist_err(target=target_str)
    dist_method = catalog_access.get_target_dist_method(target=target_str)

    delta_ms_list.append(delta_ms)
    dist_list.append(dist)
    dist_err_list.append(dist_err)
    dist_method_list.append(dist_method)
    frac_gc_list_hum.append(sum(mask_gc_hum) / sum(detectable_hum))
    frac_cascade_list_hum.append(sum(mask_cascade_hum) / sum(detectable_hum))
    frac_young_list_hum.append(sum(mask_young_hum) / sum(detectable_hum))
    frac_no_class_list_hum.append(sum(mask_no_class_hum) / sum(detectable_hum))
    frac_gc_list_ml.append(sum(mask_gc_ml) / sum(detectable_ml))
    frac_cascade_list_ml.append(sum(mask_cascade_ml) / sum(detectable_ml))
    frac_young_list_ml.append(sum(mask_young_ml) / sum(detectable_ml))
    frac_no_class_list_ml.append(sum(mask_no_class_ml) / sum(detectable_ml))



# get pearson coeeficients
pearson_gc_hum = stats.pearsonr(delta_ms_list, frac_gc_list_hum).statistic
pearson_cascade_hum = stats.pearsonr(delta_ms_list, frac_cascade_list_hum).statistic
pearson_young_hum = stats.pearsonr(delta_ms_list, frac_young_list_hum).statistic
pearson_no_class_hum = stats.pearsonr(delta_ms_list, frac_no_class_list_hum).statistic

pearson_gc_ml = stats.pearsonr(delta_ms_list, frac_gc_list_ml).statistic
pearson_cascade_ml = stats.pearsonr(delta_ms_list, frac_cascade_list_ml).statistic
pearson_young_ml = stats.pearsonr(delta_ms_list, frac_young_list_ml).statistic
pearson_no_class_ml = stats.pearsonr(delta_ms_list, frac_no_class_list_ml).statistic


delta_ms_list = np.array(delta_ms_list)

dist_list = np.array(dist_list)
dist_err_list = np.array(dist_err_list)
dist_method_list = np.array(dist_method_list)

frac_gc_list_hum = np.array(frac_gc_list_hum)
frac_cascade_list_hum = np.array(frac_cascade_list_hum)
frac_young_list_hum = np.array(frac_young_list_hum)
frac_no_class_list_hum = np.array(frac_no_class_list_hum)

frac_gc_list_ml = np.array(frac_gc_list_ml)
frac_cascade_list_ml = np.array(frac_cascade_list_ml)
frac_young_list_ml = np.array(frac_young_list_ml)
frac_no_class_list_ml = np.array(frac_no_class_list_ml)


mask_accurate_dist = (dist_method_list == 'TRGB') | (dist_method_list == 'Cepheid') | (dist_method_list == 'Mira')

fig, ax = plt.subplots(ncols=4, nrows=2, sharex='all', sharey='row', figsize=(30, 17))
fontsize = 30

ax[0, 0].scatter(dist_list[mask_accurate_dist], frac_young_list_hum[mask_accurate_dist], s=200, color='tab:blue')
ax[0, 1].scatter(dist_list[mask_accurate_dist], frac_cascade_list_hum[mask_accurate_dist], s=200, color='tab:green')
ax[0, 2].scatter(dist_list[mask_accurate_dist], frac_gc_list_hum[mask_accurate_dist], s=200, color='tab:red')
ax[0, 3].scatter(dist_list[mask_accurate_dist], frac_no_class_list_hum[mask_accurate_dist], s=200, color='tab:gray')

ax[0, 0].scatter(dist_list[np.invert(mask_accurate_dist)], frac_young_list_hum[np.invert(mask_accurate_dist)], s=200, linewidth=4, facecolor='None', color='tab:blue')
ax[0, 1].scatter(dist_list[np.invert(mask_accurate_dist)], frac_cascade_list_hum[np.invert(mask_accurate_dist)], s=200, linewidth=4, facecolor='None', color='tab:green')
ax[0, 2].scatter(dist_list[np.invert(mask_accurate_dist)], frac_gc_list_hum[np.invert(mask_accurate_dist)], s=200, linewidth=4, facecolor='None', color='tab:red')
ax[0, 3].scatter(dist_list[np.invert(mask_accurate_dist)], frac_no_class_list_hum[np.invert(mask_accurate_dist)], s=200, linewidth=4, facecolor='None', color='tab:gray')



ax[1, 0].scatter(dist_list[mask_accurate_dist], frac_young_list_ml[mask_accurate_dist], s=200, color='tab:blue')
ax[1, 1].scatter(dist_list[mask_accurate_dist], frac_cascade_list_ml[mask_accurate_dist], s=200, color='tab:green')
ax[1, 2].scatter(dist_list[mask_accurate_dist], frac_gc_list_ml[mask_accurate_dist], s=200, color='tab:red')
ax[1, 3].scatter(dist_list[mask_accurate_dist], frac_no_class_list_ml[mask_accurate_dist], s=200, color='tab:gray')

ax[1, 0].scatter(dist_list[np.invert(mask_accurate_dist)], frac_young_list_ml[np.invert(mask_accurate_dist)], s=200, linewidth=4, facecolor='None', color='tab:blue')
ax[1, 1].scatter(dist_list[np.invert(mask_accurate_dist)], frac_cascade_list_ml[np.invert(mask_accurate_dist)], s=200, linewidth=4, facecolor='None', color='tab:green')
ax[1, 2].scatter(dist_list[np.invert(mask_accurate_dist)], frac_gc_list_ml[np.invert(mask_accurate_dist)], s=200, linewidth=4, facecolor='None', color='tab:red')
ax[1, 3].scatter(dist_list[np.invert(mask_accurate_dist)], frac_no_class_list_ml[np.invert(mask_accurate_dist)], s=200, linewidth=4, facecolor='None', color='tab:gray')

# linear fit
dummy_x = np.linspace(np.min(dist_list), np.max(dist_list), 10)

coef_cascade_hum, cov_cascade_hum = np.polyfit(dist_list, frac_cascade_list_hum,1, cov=True)
coef_cascade_ml, cov_cascade_ml = np.polyfit(dist_list, frac_cascade_list_ml,1, cov=True)
poly1d_fn_cascade_hum = np.poly1d(coef_cascade_hum)
poly1d_fn_cascade_ml = np.poly1d(coef_cascade_ml)
uncert_cascade_hum = np.sqrt(np.diag(cov_cascade_hum))
uncert_cascade_ml = np.sqrt(np.diag(cov_cascade_ml))

print('coef_cascade_hum ', coef_cascade_hum)
print('coef_cascade_ml ', coef_cascade_ml)
print('uncert_cascade_hum ', uncert_cascade_hum)
print('uncert_cascade_ml ', uncert_cascade_ml)


# ax[0, 1].plot(dummy_x, poly1d_fn_cascade_hum(dummy_x), color='k', linestyle='--', linewidth=3,
#               label=r'%.2f$(\pm %.2f)\times \Delta$MS + %.2f($\pm$%.2f)' %
#                     (coef_cascade_hum[0], uncert_cascade_hum[0], coef_cascade_hum[1], uncert_cascade_hum[1]))
# ax[1, 1].plot(dummy_x, poly1d_fn_cascade_ml(dummy_x), color='k', linestyle='--', linewidth=3,
#               label=r'%.2f$(\pm %.2f)\times \Delta$MS + %.2f($\pm$%.2f)' %
#                     (coef_cascade_ml[0], uncert_cascade_ml[0], coef_cascade_ml[1], uncert_cascade_ml[1]))
# ax[0, 1].legend(frameon=True, loc=4, fontsize=fontsize-6)
# ax[1, 1].legend(frameon=True, loc=4, fontsize=fontsize-6)

ax[0, 0].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_young_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 0].transAxes)
ax[0, 1].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_cascade_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 1].transAxes)
ax[0, 2].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_gc_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 2].transAxes)
ax[0, 3].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_no_class_hum, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[0, 3].transAxes)

ax[1, 0].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_young_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 0].transAxes)
ax[1, 1].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_cascade_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 1].transAxes)
ax[1, 2].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_gc_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 2].transAxes)
ax[1, 3].text(0.97, 0.97, 'Pearson Coeff. = %.2f' % pearson_no_class_ml, horizontalalignment='right', verticalalignment='top', fontsize=fontsize-2, transform=ax[1, 3].transAxes)


ax[0, 0].set_ylabel('Number Fraction of Clusters', fontsize=fontsize)
ax[1, 0].set_ylabel('Number Fraction of Clusters', fontsize=fontsize)

ax[1, 0].set_xlabel(r'Dist [Mpc]', fontsize=fontsize)
ax[1, 1].set_xlabel(r'Dist [Mpc]', fontsize=fontsize)
ax[1, 2].set_xlabel(r'Dist [Mpc]', fontsize=fontsize)
ax[1, 3].set_xlabel(r'Dist [Mpc]', fontsize=fontsize)

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

fig.subplots_adjust(left=0.04, bottom=0.06, right=0.995, top=0.97, wspace=0.01, hspace=0.01)
# plt.tight_layout()
fig.savefig('plot_output/ms_stats_dist.png')
fig.savefig('plot_output/ms_stats_dist.pdf')
