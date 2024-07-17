import numpy as np
import scipy.stats

import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from scipy.stats import gaussian_kde, spearmanr
from scipy import stats

from xgaltool import analysis_tools, plotting_tools
import matplotlib

from photometry_tools.plotting_tools import DensityContours
from matplotlib import cm
from matplotlib.patches import ConnectionPatch
from scipy.spatial import ConvexHull


morph_dict = {
    'ngc1365': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc1672': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc4303': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc7496': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},

    'ngc1385': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc1559': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc4536': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc4254': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},

    'ngc4654': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc1087': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc1097': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc1792': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': False, 'quiescent': False},

    'ngc1566': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
    'ngc2835': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc5248': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc2903': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},

    'ngc4321': {'sf_bar': False, 'cent_ring': True, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc3627': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
    'ngc0628': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
    'ngc4535': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},

    'ngc3621': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
    'ngc6744': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
    'ngc3351': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
    'ngc5068': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},

    'ic5332': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ic1954': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc4298': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc1300': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},

    'ngc1512': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
    'ngc0685': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc4569': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': True},
    'ngc1433': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},

    'ngc4689': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc4571': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc1317': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc4548': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': True, 'flocc': False, 'quiescent': False},

    'ngc2775': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': True},
    'ngc4826': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': False, 'quiescent': True},
}

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
target_list = np.array(target_list, dtype=str)

catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc)
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml', cluster_class='class3')


delta_ms_list = []

p20_age_list_hum = []
mean_age_list_hum = []
p80_age_list_hum = []
mean_age_mass_weighted_list_hum = []

p20_age_list_ml = []
mean_age_list_ml = []
p80_age_list_ml = []
mean_age_mass_weighted_list_ml = []

mask_bar_sf = []
mask_global_arms = []
mask_bulge = []
mask_flocculent = []

for index in range(len(target_list)):
    target = target_list[index]

    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    mass_hum_12 = catalog_access.get_hst_cc_stellar_m(target=target)

    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    mass_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')


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


    good_values_hum_12 = np.invert(np.isnan(np.log10(age_hum_12)))
    p20_age_list_hum.append(np.nanpercentile(np.log10(age_hum_12[good_values_hum_12]) + 6, 20))
    mean_age_list_hum.append(np.nanmean(np.log10(age_hum_12[good_values_hum_12]) + 6))
    p80_age_list_hum.append(np.nanpercentile(np.log10(age_hum_12[good_values_hum_12]) + 6, 80))
    mean_age_mass_weighted_list_hum.append(np.average(np.log10(age_hum_12[good_values_hum_12]) + 6, weights=mass_hum_12[good_values_hum_12]))

    good_values_ml_12 = np.invert(np.isnan(np.log10(age_ml_12)))
    p20_age_list_ml.append(np.nanpercentile(np.log10(age_ml_12[good_values_ml_12]) + 6, 20))
    mean_age_list_ml.append(np.nanmean(np.log10(age_ml_12[good_values_ml_12]) + 6))
    p80_age_list_ml.append(np.nanpercentile(np.log10(age_ml_12[good_values_ml_12]) + 6, 80))
    mean_age_mass_weighted_list_ml.append(np.average(np.log10(age_ml_12[good_values_ml_12]) + 6, weights=mass_ml_12[good_values_ml_12]))

    mask_bar_sf.append(morph_dict[target_str]['sf_bar'] + morph_dict[target_str]['sf_end_bars'])
    mask_global_arms.append(morph_dict[target_str]['glob_arms'])
    mask_bulge.append(morph_dict[target_str]['bulge'])
    mask_flocculent.append(morph_dict[target_str]['flocc'] + morph_dict[target_str]['quiescent'])

print(mean_age_mass_weighted_list_hum)

print(mean_age_mass_weighted_list_ml)

delta_ms_list = np.array(delta_ms_list, dtype=float)

p20_age_list_hum = np.array(p20_age_list_hum, dtype=float)
mean_age_list_hum = np.array(mean_age_list_hum, dtype=float)
p80_age_list_hum = np.array(p80_age_list_hum, dtype=float)
mean_age_mass_weighted_list_hum = np.array(mean_age_mass_weighted_list_hum, dtype=float)
p20_age_list_ml = np.array(p20_age_list_ml, dtype=float)
mean_age_list_ml = np.array(mean_age_list_ml, dtype=float)
p80_age_list_ml = np.array(p80_age_list_ml, dtype=float)
mean_age_mass_weighted_list_ml = np.array(mean_age_mass_weighted_list_ml, dtype=float)

mask_bar_sf = np.array(mask_bar_sf, dtype=bool)
mask_global_arms = np.array(mask_global_arms, dtype=bool)
mask_bulge = np.array(mask_bulge, dtype=bool)
mask_flocculent = np.array(mask_flocculent, dtype=bool)
mask_not_classified = np.invert(mask_bar_sf + mask_global_arms + mask_bulge + mask_flocculent)


spearman_coeff_ms_age_hum, p_val_ms_age_hum = spearmanr(delta_ms_list, mean_age_list_hum)
spearman_coeff_ms_age_ml, p_val_ms_age_ml = spearmanr(delta_ms_list, mean_age_list_ml)
spearman_coeff_ms_age_mass_weighted_hum, p_val_ms_age_mass_weighted_hum = spearmanr(delta_ms_list, mean_age_mass_weighted_list_hum)
spearman_coeff_ms_age_mass_weighted_ml, p_val_ms_age_mass_weighted_ml = spearmanr(delta_ms_list, mean_age_mass_weighted_list_ml)

pearson_coeff_ms_age_hum = stats.pearsonr(delta_ms_list, mean_age_list_hum).statistic
pearson_coeff_ms_age_ml = stats.pearsonr(delta_ms_list, mean_age_list_ml).statistic
pearson_coeff_ms_age_mass_weighted_hum = stats.pearsonr(delta_ms_list, mean_age_mass_weighted_list_hum).statistic
pearson_coeff_ms_age_mass_weighted_ml = stats.pearsonr(delta_ms_list, mean_age_mass_weighted_list_ml).statistic


fig, ax = plt.subplots(ncols=2, nrows=2, sharex='all', sharey='all', figsize=(35, 25))
fontsize = 40


ax[0, 0].scatter(delta_ms_list[mask_bar_sf], mean_age_list_hum[mask_bar_sf],
                 marker='s', color='tab:blue', facecolor='None', linewidth=3, s=1200, label='SF Bar')
ax[0, 0].scatter(delta_ms_list[mask_global_arms], mean_age_list_hum[mask_global_arms],
                 marker='o', color='tab:orange', facecolor='None', linewidth=3, s=800, label='Global Arms')
ax[0, 0].scatter(delta_ms_list[mask_bulge], mean_age_list_hum[mask_bulge],
                 marker='*', color='tab:green', facecolor='None', linewidth=3, s=900, label='Bulge')
ax[0, 0].scatter(delta_ms_list[mask_flocculent], mean_age_list_hum[mask_flocculent],
                 marker='d', color='tab:red', s=300, label='Quiescent + Flocculent')
ax[0, 0].scatter(delta_ms_list[mask_not_classified], mean_age_list_hum[mask_not_classified], color='gray', s=400)


ax[0, 1].scatter(delta_ms_list[mask_bar_sf], mean_age_list_ml[mask_bar_sf],
                 marker='s', color='tab:blue', facecolor='None', linewidth=3, s=1200)
ax[0, 1].scatter(delta_ms_list[mask_global_arms], mean_age_list_ml[mask_global_arms],
                 marker='o', color='tab:orange', facecolor='None', linewidth=3, s=800)
ax[0, 1].scatter(delta_ms_list[mask_bulge], mean_age_list_ml[mask_bulge],
                 marker='*', color='tab:green', facecolor='None', linewidth=3, s=600)
ax[0, 1].scatter(delta_ms_list[mask_flocculent], mean_age_list_ml[mask_flocculent],
                 marker='d', color='tab:red', s=300)
ax[0, 1].scatter(delta_ms_list[mask_not_classified], mean_age_list_ml[mask_not_classified], color='gray', s=400)


ax[1, 0].scatter(delta_ms_list[mask_bar_sf], mean_age_mass_weighted_list_hum[mask_bar_sf],
                 marker='s', color='tab:blue', facecolor='None', linewidth=3, s=1200, label='SF Bar')
ax[1, 0].scatter(delta_ms_list[mask_global_arms], mean_age_mass_weighted_list_hum[mask_global_arms],
                 marker='o', color='tab:orange', facecolor='None', linewidth=3, s=800, label='Global Arms')
ax[1, 0].scatter(delta_ms_list[mask_bulge], mean_age_mass_weighted_list_hum[mask_bulge],
                 marker='*', color='tab:green', facecolor='None', linewidth=3, s=900, label='Bulge')
ax[1, 0].scatter(delta_ms_list[mask_flocculent], mean_age_mass_weighted_list_hum[mask_flocculent],
                 marker='d', color='tab:red', s=300, label='Quiescent + Flocculent')
ax[1, 0].scatter(delta_ms_list[mask_not_classified], mean_age_mass_weighted_list_hum[mask_not_classified], color='gray', s=400)


ax[1, 1].scatter(delta_ms_list[mask_bar_sf], mean_age_mass_weighted_list_ml[mask_bar_sf],
                 marker='s', color='tab:blue', facecolor='None', linewidth=3, s=1200, label='SF Bar')
ax[1, 1].scatter(delta_ms_list[mask_global_arms], mean_age_mass_weighted_list_ml[mask_global_arms],
                 marker='o', color='tab:orange', facecolor='None', linewidth=3, s=800, label='Global Arms')
ax[1, 1].scatter(delta_ms_list[mask_bulge], mean_age_mass_weighted_list_ml[mask_bulge],
                 marker='*', color='tab:green', facecolor='None', linewidth=3, s=900, label='Bulge')
ax[1, 1].scatter(delta_ms_list[mask_flocculent], mean_age_mass_weighted_list_ml[mask_flocculent],
                 marker='d', color='tab:red', s=300, label='Quiescent + Flocculent')
ax[1, 1].scatter(delta_ms_list[mask_not_classified], mean_age_mass_weighted_list_ml[mask_not_classified], color='gray', s=400)

# 1599, 628, 2775, 4826
ax[1, 0].annotate('NGC1365', xycoords='data', fontsize=fontsize-5,
                  xy=(delta_ms_list[target_list == 'ngc1365'],
                      mean_age_mass_weighted_list_hum[target_list == 'ngc1365']),
                  xytext=(delta_ms_list[target_list == 'ngc1365'] - 0.2,
                          mean_age_mass_weighted_list_hum[target_list == 'ngc1365'] + 0.8),
                  textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

ax[1, 0].annotate('NGC4826', xycoords='data', fontsize=fontsize-5,
                  xy=(delta_ms_list[target_list == 'ngc4826'],
                      mean_age_mass_weighted_list_hum[target_list == 'ngc4826']),
                  xytext=(delta_ms_list[target_list == 'ngc4826'] + 0.05,
                          mean_age_mass_weighted_list_hum[target_list == 'ngc4826'] - 1.5),
                  textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

ax[1, 0].annotate('NGC2775', xycoords='data', fontsize=fontsize-5,
                  xy=(delta_ms_list[target_list == 'ngc2775'],
                      mean_age_mass_weighted_list_hum[target_list == 'ngc2775']),
                  xytext=(delta_ms_list[target_list == 'ngc2775'] + 0.025,
                          mean_age_mass_weighted_list_hum[target_list == 'ngc2775'] - 1.1),
                  textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

ax[1, 0].annotate('NGC1566', xycoords='data', fontsize=fontsize-5,
                  xy=(delta_ms_list[target_list == 'ngc1566'],
                      mean_age_mass_weighted_list_hum[target_list == 'ngc1566']),
                  xytext=(delta_ms_list[target_list == 'ngc1566'] + 0.0,
                          mean_age_mass_weighted_list_hum[target_list == 'ngc1566'] + 0.6),
                  textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

ax[1, 0].annotate('NGC628C', xycoords='data', fontsize=fontsize-5,
                  xy=(delta_ms_list[target_list == 'ngc0628c'],
                      mean_age_mass_weighted_list_hum[target_list == 'ngc0628c']),
                  xytext=(delta_ms_list[target_list == 'ngc0628c'] - 0.1,
                          mean_age_mass_weighted_list_hum[target_list == 'ngc0628c'] + 0.15),
                  textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))


ax[0, 0].legend(frameon=True, loc=1, fontsize=fontsize - 5)


ax[0, 0].set_ylabel(r'Mean log(Age/yr)', fontsize=fontsize)
ax[1, 0].set_ylabel(r'Mean log(Age/yr) (mass weighted)', fontsize=fontsize)

ax[1, 0].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)
ax[1, 1].set_xlabel(r'$\Delta$MS [dex]', fontsize=fontsize)

ax[0, 0].text(0.03, 0.97, 'Hum C1+C2', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 0].transAxes)
ax[0, 1].text(0.03, 0.97, 'ML C1+C2', horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 1].transAxes)
ax[1, 0].text(0.03, 0.13, 'Hum C1+C2', horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize,
              transform=ax[1, 0].transAxes)
ax[1, 1].text(0.03, 0.13, 'ML C1+C2', horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize,
              transform=ax[1, 1].transAxes)

ax[0, 0].text(0.03, 0.92, 'Spearman coeff. = %.2f' % spearman_coeff_ms_age_hum,
              horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 0].transAxes)
ax[0, 1].text(0.03, 0.92, 'Spearman coeff. = %.2f' % spearman_coeff_ms_age_ml,
              horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 1].transAxes)
ax[1, 0].text(0.03, 0.08, 'Spearman coeff. = %.2f' % spearman_coeff_ms_age_mass_weighted_hum,
              horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize,
              transform=ax[1, 0].transAxes)
ax[1, 1].text(0.03, 0.08, 'Spearman coeff. = %.2f' % spearman_coeff_ms_age_mass_weighted_ml,
              horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize,
              transform=ax[1, 1].transAxes)

ax[0, 0].text(0.03, 0.87, 'Pearson coeff. = %.2f' % pearson_coeff_ms_age_hum,
              horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 0].transAxes)
ax[0, 1].text(0.03, 0.87, 'Pearson coeff. = %.2f' % pearson_coeff_ms_age_ml,
              horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
              transform=ax[0, 1].transAxes)
ax[1, 0].text(0.03, 0.03, 'Pearson coeff. = %.2f' % pearson_coeff_ms_age_mass_weighted_hum,
              horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize,
              transform=ax[1, 0].transAxes)
ax[1, 1].text(0.03, 0.03, 'Pearson coeff. = %.2f' % pearson_coeff_ms_age_mass_weighted_ml,
              horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize,
              transform=ax[1, 1].transAxes)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


fig.subplots_adjust(left=0.06, bottom=0.05, right=0.995, top=0.995, wspace=0.01, hspace=0.01)
# plt.tight_layout()
fig.savefig('plot_output/ms_age_stats.png')
fig.savefig('plot_output/ms_age_stats.pdf')
exit()

