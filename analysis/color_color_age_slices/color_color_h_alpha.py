import os.path

import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from scipy.stats import gaussian_kde
import dust_tools.extinction_tools
from matplotlib.colorbar import ColorbarBase
import matplotlib

model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access_ir4 = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path,
                                                                hst_cc_ver='IR4')
catalog_access_fix = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path,
                                                                hst_cc_ver='SEDfix_Ha1_inclusiveGCcc_inclusiveGCclass')

target_list = catalog_access_ir4.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access_ir4.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access_ir4.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access_ir4.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access_ir4.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access_ir4.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

catalog_access_fix.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access_fix.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access_fix.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access_fix.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

extra_file_path = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/'

target_name_hum = np.array([], dtype=str)
clcl_color_hum = np.array([])
color_vi_hum = np.array([])
color_ub_hum = np.array([])
detect_u_hum = np.array([], dtype=bool)
detect_b_hum = np.array([], dtype=bool)
detect_v_hum = np.array([], dtype=bool)
detect_i_hum = np.array([], dtype=bool)
age_ir4_hum = np.array([])
age_fix_hum = np.array([])
age_fix_youngest_hum = np.array([])
age_fix_likeliest_hum = np.array([])
h_alpha_intensity_hum = np.array([])

target_name_ml = np.array([], dtype=str)
clcl_color_ml = np.array([])
color_vi_ml = np.array([])
color_ub_ml = np.array([])
detect_u_ml = np.array([], dtype=bool)
detect_b_ml = np.array([], dtype=bool)
detect_v_ml = np.array([], dtype=bool)
detect_i_ml = np.array([], dtype=bool)
age_ir4_ml = np.array([])
age_fix_ml = np.array([])
age_fix_youngest_ml = np.array([])
age_fix_likeliest_ml = np.array([])
h_alpha_intensity_ml = np.array([])


for index in range(len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    if 'F438W' in catalog_access_ir4.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'

    cluster_class_hum_12 = catalog_access_ir4.get_hst_cc_class_human(target=target)
    color_vi_hum_12 = catalog_access_ir4.get_hst_color_vi_vega(target=target)
    color_ub_hum_12 = catalog_access_ir4.get_hst_color_ub_vega(target=target)

    detect_u_hum_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, band='F336W') > 0
    detect_b_hum_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, band=b_band) > 0
    detect_v_hum_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, band='F555W') > 0
    detect_i_hum_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, band='F814W') > 0

    age_ir4_hum_12 = catalog_access_ir4.get_hst_cc_age(target=target)
    age_fix_hum_12 = catalog_access_fix.get_hst_cc_age(target=target)
    age_fix_youngest_hum_12 = catalog_access_fix.get_hst_cc_age_young_mode(target=target)
    age_fix_likeliest_hum_12 = catalog_access_fix.get_hst_cc_age_likely_mode(target=target)
    h_alpha_intensity_hum_12 = catalog_access_fix.get_h_alpha_medsub(target=target)

    cluster_class_hum_3 = catalog_access_ir4.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access_ir4.get_hst_color_vi_vega(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access_ir4.get_hst_color_ub_vega(target=target, cluster_class='class3')

    detect_u_hum_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F336W') > 0
    detect_b_hum_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) > 0
    detect_v_hum_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') > 0
    detect_i_hum_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F814W') > 0

    age_ir4_hum_3 = catalog_access_ir4.get_hst_cc_age(target=target, cluster_class='class3')
    age_fix_hum_3 = catalog_access_fix.get_hst_cc_age(target=target, cluster_class='class3')
    age_fix_youngest_hum_3 = catalog_access_fix.get_hst_cc_age_young_mode(target=target, cluster_class='class3')
    age_fix_likeliest_hum_3 = catalog_access_fix.get_hst_cc_age_likely_mode(target=target, cluster_class='class3')
    h_alpha_intensity_hum_3 = catalog_access_fix.get_h_alpha_medsub(target=target, cluster_class='class3')


    color_vi_hum = np.concatenate([color_vi_hum, color_vi_hum_12, color_vi_hum_3])
    color_ub_hum = np.concatenate([color_ub_hum, color_ub_hum_12, color_ub_hum_3])
    detect_u_hum = np.concatenate([detect_u_hum, detect_u_hum_12, detect_u_hum_3])
    detect_b_hum = np.concatenate([detect_b_hum, detect_b_hum_12, detect_b_hum_3])
    detect_v_hum = np.concatenate([detect_v_hum, detect_v_hum_12, detect_v_hum_3])
    detect_i_hum = np.concatenate([detect_i_hum, detect_i_hum_12, detect_i_hum_3])
    clcl_color_hum = np.concatenate([clcl_color_hum, cluster_class_hum_12, cluster_class_hum_3])
    target_name_hum_12 = np.array([target]*len(cluster_class_hum_12))
    target_name_hum_3 = np.array([target]*len(cluster_class_hum_3))
    target_name_hum = np.concatenate([target_name_hum, target_name_hum_12, target_name_hum_3])
    age_ir4_hum = np.concatenate([age_ir4_hum, age_ir4_hum_12, age_ir4_hum_3])
    age_fix_hum = np.concatenate([age_fix_hum, age_fix_hum_12, age_fix_hum_3])
    age_fix_youngest_hum = np.concatenate([age_fix_youngest_hum, age_fix_youngest_hum_12, age_fix_youngest_hum_3])
    age_fix_likeliest_hum = np.concatenate([age_fix_likeliest_hum, age_fix_likeliest_hum_12, age_fix_likeliest_hum_3])
    h_alpha_intensity_hum = np.concatenate([h_alpha_intensity_hum, h_alpha_intensity_hum_12, h_alpha_intensity_hum_3])


    cluster_class_ml_12 = catalog_access_ir4.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    color_vi_ml_12 = catalog_access_ir4.get_hst_color_vi_vega(target=target, classify='ml')
    color_ub_ml_12 = catalog_access_ir4.get_hst_color_ub_vega(target=target, classify='ml')

    detect_u_ml_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0
    detect_b_ml_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0
    detect_v_ml_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0
    detect_i_ml_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0

    age_ir4_ml_12 = catalog_access_ir4.get_hst_cc_age(target=target, classify='ml')
    age_fix_ml_12 = catalog_access_fix.get_hst_cc_age(target=target, classify='ml')
    age_fix_youngest_ml_12 = catalog_access_fix.get_hst_cc_age_young_mode(target=target, classify='ml')
    age_fix_likeliest_ml_12 = catalog_access_fix.get_hst_cc_age_likely_mode(target=target, classify='ml')
    h_alpha_intensity_ml_12 = catalog_access_fix.get_h_alpha_medsub(target=target, classify='ml')

    cluster_class_ml_3 = catalog_access_ir4.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access_ir4.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access_ir4.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')

    detect_u_ml_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F336W') > 0
    detect_b_ml_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) > 0
    detect_v_ml_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') > 0
    detect_i_ml_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F814W') > 0

    age_ir4_ml_3 = catalog_access_ir4.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    age_fix_ml_3 = catalog_access_fix.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    age_fix_youngest_ml_3 = catalog_access_fix.get_hst_cc_age_young_mode(target=target, classify='ml', cluster_class='class3')
    age_fix_likeliest_ml_3 = catalog_access_fix.get_hst_cc_age_likely_mode(target=target, classify='ml', cluster_class='class3')
    h_alpha_intensity_ml_3 = catalog_access_fix.get_h_alpha_medsub(target=target, classify='ml', cluster_class='class3')


    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    detect_u_ml = np.concatenate([detect_u_ml, detect_u_ml_12, detect_u_ml_3])
    detect_b_ml = np.concatenate([detect_b_ml, detect_b_ml_12, detect_b_ml_3])
    detect_v_ml = np.concatenate([detect_v_ml, detect_v_ml_12, detect_v_ml_3])
    detect_i_ml = np.concatenate([detect_i_ml, detect_i_ml_12, detect_i_ml_3])
    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    target_name_ml_12 = np.array([target]*len(cluster_class_ml_12))
    target_name_ml_3 = np.array([target]*len(cluster_class_ml_3))
    target_name_ml = np.concatenate([target_name_ml, target_name_ml_12, target_name_ml_3])
    age_ir4_ml = np.concatenate([age_ir4_ml, age_ir4_ml_12, age_ir4_ml_3])
    age_fix_ml = np.concatenate([age_fix_ml, age_fix_ml_12, age_fix_ml_3])
    age_fix_youngest_ml = np.concatenate([age_fix_youngest_ml, age_fix_youngest_ml_12, age_fix_youngest_ml_3])
    age_fix_likeliest_ml = np.concatenate([age_fix_likeliest_ml, age_fix_likeliest_ml_12, age_fix_likeliest_ml_3])
    h_alpha_intensity_ml = np.concatenate([h_alpha_intensity_ml, h_alpha_intensity_ml_12, h_alpha_intensity_ml_3])


class_12_hum = (clcl_color_hum == 1) | (clcl_color_hum == 2)
class_12_ml = (clcl_color_ml == 1) | (clcl_color_ml == 2)




age_groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21]

fig_h_alpha_hum, ax_h_alpha_hum = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fig_youngest_hum, ax_youngest_hum = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fig_likeliest_hum, ax_likeliest_hum = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fig_h_alpha_ml, ax_h_alpha_ml = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fig_youngest_ml, ax_youngest_ml = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fig_likeliest_ml, ax_likeliest_ml = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18

cmap_h_alpha = matplotlib.cm.get_cmap('rainbow')
norm_h_alpha = matplotlib.colors.Normalize(vmin=0, vmax=5)

cmap_age_young = matplotlib.cm.get_cmap('rainbow')
norm_age_young = matplotlib.colors.Normalize(vmin=0, vmax=5)

cmap_age_likeliest = matplotlib.cm.get_cmap('rainbow')
norm_age_likeliest = matplotlib.colors.Normalize(vmin=0, vmax=5)

vi_int = 0.8
ub_int = -2.2
av_value = 1

x_lim_vi = (-1.0, 2.3)
y_lim_ub = (1.25, -2.8)


row_index = 0
col_index = 0
for index, age in enumerate(age_groups):
    age_mask_hum = age_ir4_hum == age
    age_mask_ml = age_ir4_ml == age

    ax_h_alpha_hum[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)
    ax_youngest_hum[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)
    ax_likeliest_hum[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)

    ax_h_alpha_ml[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)
    ax_youngest_ml[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)
    ax_likeliest_ml[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)

    ax_h_alpha_hum[row_index, col_index].scatter(color_vi_hum[class_12_hum * age_mask_hum], color_ub_hum[class_12_hum * age_mask_hum],
                                                 c=10**(h_alpha_intensity_hum[class_12_hum * age_mask_hum]-6), norm=norm_h_alpha, cmap=cmap_h_alpha, s=10)
    ax_h_alpha_ml[row_index, col_index].scatter(color_vi_ml[class_12_ml * age_mask_ml], color_ub_ml[class_12_ml * age_mask_ml],
                                                 c=10**(h_alpha_intensity_ml[class_12_ml * age_mask_ml]-6), norm=norm_h_alpha, cmap=cmap_h_alpha, s=10)

    ax_youngest_hum[row_index, col_index].scatter(color_vi_hum[class_12_hum * age_mask_hum], color_ub_hum[class_12_hum * age_mask_hum],
                                                 c=10**(age_fix_youngest_hum[class_12_hum * age_mask_hum] - 6), norm=norm_age_young, cmap=cmap_age_young, s=10)
    ax_youngest_ml[row_index, col_index].scatter(color_vi_ml[class_12_ml * age_mask_ml], color_ub_ml[class_12_ml * age_mask_ml],
                                                 c=10**(age_fix_youngest_ml[class_12_ml * age_mask_ml] - 6), norm=norm_age_young, cmap=cmap_age_young, s=10)

    ax_likeliest_hum[row_index, col_index].scatter(color_vi_hum[class_12_hum * age_mask_hum], color_ub_hum[class_12_hum * age_mask_hum],
                                                 c=age_fix_likeliest_hum[class_12_hum * age_mask_hum], norm=norm_age_likeliest, cmap=cmap_age_likeliest, s=10)
    ax_likeliest_ml[row_index, col_index].scatter(color_vi_ml[class_12_ml * age_mask_ml], color_ub_ml[class_12_ml * age_mask_ml],
                                                 c=age_fix_likeliest_ml[class_12_ml * age_mask_ml], norm=norm_age_likeliest, cmap=cmap_age_likeliest, s=10)


    ax_h_alpha_hum[row_index, col_index].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              '%i Myr' % age, horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize+5)
    ax_youngest_hum[row_index, col_index].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              '%i Myr' % age, horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize+5)
    ax_likeliest_hum[row_index, col_index].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              '%i Myr' % age, horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize+5)
    ax_h_alpha_ml[row_index, col_index].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              '%i Myr' % age, horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize+5)
    ax_youngest_ml[row_index, col_index].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              '%i Myr' % age, horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize+5)
    ax_likeliest_ml[row_index, col_index].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              '%i Myr' % age, horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize+5)

    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0


ax_h_alpha_hum[4, 3].axis('off')
ax_youngest_hum[4, 3].axis('off')
ax_likeliest_hum[4, 3].axis('off')
ax_h_alpha_ml[4, 3].axis('off')
ax_youngest_ml[4, 3].axis('off')
ax_likeliest_ml[4, 3].axis('off')

ax_h_alpha_hum_cbar = fig_h_alpha_hum.add_axes([0.82, 0.13, 0.015, 0.1])
ax_youngest_hum_cbar = fig_youngest_hum.add_axes([0.82, 0.13, 0.015, 0.1])
ax_likeliest_hum_cbar = fig_likeliest_hum.add_axes([0.82, 0.13, 0.015, 0.1])
ax_h_alpha_ml_cbar = fig_h_alpha_ml.add_axes([0.82, 0.13, 0.015, 0.1])
ax_youngest_ml_cbar = fig_youngest_ml.add_axes([0.82, 0.13, 0.015, 0.1])
ax_likeliest_ml_cbar = fig_likeliest_ml.add_axes([0.82, 0.13, 0.015, 0.1])

ColorbarBase(ax_h_alpha_hum_cbar, orientation='vertical', cmap=cmap_h_alpha, norm=norm_h_alpha, extend='neither', ticks=None)
ColorbarBase(ax_h_alpha_ml_cbar, orientation='vertical', cmap=cmap_h_alpha, norm=norm_h_alpha, extend='neither', ticks=None)
ColorbarBase(ax_youngest_hum_cbar, orientation='vertical', cmap=cmap_age_young, norm=norm_age_young, extend='neither', ticks=None)
ColorbarBase(ax_youngest_ml_cbar, orientation='vertical', cmap=cmap_age_young, norm=norm_age_young, extend='neither', ticks=None)
ColorbarBase(ax_likeliest_hum_cbar, orientation='vertical', cmap=cmap_age_likeliest, norm=norm_age_likeliest, extend='neither', ticks=None)
ColorbarBase(ax_likeliest_ml_cbar, orientation='vertical', cmap=cmap_age_likeliest, norm=norm_age_likeliest, extend='neither', ticks=None)

ax_h_alpha_hum_cbar.set_ylabel(r'$\Sigma_{\rm H\alpha}$', labelpad=-4, fontsize=fontsize + 5)
ax_h_alpha_ml_cbar.set_ylabel(r'$\Sigma_{\rm H\alpha}$', labelpad=-4, fontsize=fontsize + 5)
ax_youngest_hum_cbar.set_ylabel(r'Age Myr', labelpad=-4, fontsize=fontsize + 5)
ax_youngest_ml_cbar.set_ylabel(r'Age Myr', labelpad=-4, fontsize=fontsize + 5)
ax_likeliest_hum_cbar.set_ylabel(r'Age Myr', labelpad=-4, fontsize=fontsize + 5)
ax_likeliest_ml_cbar.set_ylabel(r'Age Myr', labelpad=-4, fontsize=fontsize + 5)

ax_h_alpha_hum_cbar.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)
ax_youngest_hum_cbar.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)
ax_likeliest_hum_cbar.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)
ax_h_alpha_ml_cbar.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)
ax_youngest_ml_cbar.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)
ax_likeliest_ml_cbar.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)

ax_h_alpha_hum[0, 0].set_xlim(x_lim_vi)
ax_youngest_hum[0, 0].set_xlim(x_lim_vi)
ax_likeliest_hum[0, 0].set_xlim(x_lim_vi)
ax_h_alpha_ml[0, 0].set_xlim(x_lim_vi)
ax_youngest_ml[0, 0].set_xlim(x_lim_vi)
ax_likeliest_ml[0, 0].set_xlim(x_lim_vi)

ax_h_alpha_hum[0, 0].set_ylim(y_lim_ub)
ax_youngest_hum[0, 0].set_ylim(y_lim_ub)
ax_likeliest_hum[0, 0].set_ylim(y_lim_ub)
ax_h_alpha_ml[0, 0].set_ylim(y_lim_ub)
ax_youngest_ml[0, 0].set_ylim(y_lim_ub)
ax_likeliest_ml[0, 0].set_ylim(y_lim_ub)


fig_h_alpha_hum.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_h_alpha_hum.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_h_alpha_hum.text(0.5, 0.89, 'Class 1|2 Hum', ha='center', fontsize=fontsize)
fig_h_alpha_ml.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_h_alpha_ml.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_h_alpha_ml.text(0.5, 0.89, 'Class 1|2 ML', ha='center', fontsize=fontsize)

fig_youngest_hum.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_youngest_hum.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_youngest_hum.text(0.5, 0.89, 'Class 1|2 Hum', ha='center', fontsize=fontsize)
fig_youngest_ml.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_youngest_ml.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_youngest_ml.text(0.5, 0.89, 'Class 1|2 ML', ha='center', fontsize=fontsize)

fig_likeliest_hum.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_likeliest_hum.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_likeliest_hum.text(0.5, 0.89, 'Class 1|2 Hum', ha='center', fontsize=fontsize)
fig_likeliest_ml.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_likeliest_ml.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_likeliest_ml.text(0.5, 0.89, 'Class 1|2 ML', ha='center', fontsize=fontsize)


fig_h_alpha_hum.subplots_adjust(wspace=0, hspace=0)
fig_h_alpha_ml.subplots_adjust(wspace=0, hspace=0)
fig_youngest_hum.subplots_adjust(wspace=0, hspace=0)
fig_youngest_ml.subplots_adjust(wspace=0, hspace=0)
fig_likeliest_hum.subplots_adjust(wspace=0, hspace=0)
fig_likeliest_ml.subplots_adjust(wspace=0, hspace=0)

fig_h_alpha_hum.savefig('plot_output/ubvi_h_alpha_hum.png', bbox_inches='tight', dpi=300)
fig_h_alpha_ml.savefig('plot_output/ubvi_h_alpha_ml.png', bbox_inches='tight', dpi=300)
fig_youngest_hum.savefig('plot_output/ubvi_youngest_hum.png', bbox_inches='tight', dpi=300)
fig_youngest_ml.savefig('plot_output/ubvi_youngest_ml.png', bbox_inches='tight', dpi=300)
fig_likeliest_hum.savefig('plot_output/ubvi_likeliest_hum.png', bbox_inches='tight', dpi=300)
fig_likeliest_ml.savefig('plot_output/ubvi_likeliest_ml.png', bbox_inches='tight', dpi=300)


