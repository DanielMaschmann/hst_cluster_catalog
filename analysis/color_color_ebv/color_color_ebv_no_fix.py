import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colorbar import ColorbarBase
from photometry_tools import helper_func, data_access




cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                           hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                           morph_mask_path=morph_mask_path,
                                           sample_table_path=sample_table_path,
                                           hst_cc_ver='IR4')

# get model
model_nuvb_sol = np.load('../color_color/data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')



target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')













age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_nuvu_sol = np.load('../color_color/data_output/model_nuvu_sol.npy')
model_nuvb_sol = np.load('../color_color/data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_nuvu_sol50 = np.load('../color_color/data_output/model_nuvu_sol50.npy')
model_nuvb_sol50 = np.load('../color_color/data_output/model_nuvb_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')


color_vi_hum = np.load('../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../color_color/data_output/color_ub_hum.npy')
color_bv_hum = np.load('../color_color/data_output/color_bv_hum.npy')
color_nuvu_hum = np.load('../color_color/data_output/color_nuvu_hum.npy')
color_nuvb_hum = np.load('../color_color/data_output/color_nuvb_hum.npy')
color_vi_err_hum = np.load('../color_color/data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('../color_color/data_output/color_ub_err_hum.npy')
color_bv_err_hum = np.load('../color_color/data_output/color_bv_err_hum.npy')
color_nuvu_err_hum = np.load('../color_color/data_output/color_nuvu_err_hum.npy')
color_nuvb_err_hum = np.load('../color_color/data_output/color_nuvb_err_hum.npy')
detect_vi_hum = np.load('../color_color/data_output/detect_vi_hum.npy')
detect_ub_hum = np.load('../color_color/data_output/detect_ub_hum.npy')
detect_bv_hum = np.load('../color_color/data_output/detect_bv_hum.npy')
detect_nuvu_hum = np.load('../color_color/data_output/detect_nuvu_hum.npy')
detect_nuvb_hum = np.load('../color_color/data_output/detect_nuvb_hum.npy')
clcl_color_hum = np.load('../color_color/data_output/clcl_color_hum.npy')
age_hum = np.load('../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../color_color/data_output/ebv_hum.npy')
color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
color_bv_ml = np.load('../color_color/data_output/color_bv_ml.npy')
color_nuvu_ml = np.load('../color_color/data_output/color_nuvu_ml.npy')
color_nuvb_ml = np.load('../color_color/data_output/color_nuvb_ml.npy')
color_vi_err_ml = np.load('../color_color/data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('../color_color/data_output/color_ub_err_ml.npy')
color_bv_err_ml = np.load('../color_color/data_output/color_bv_err_ml.npy')
color_nuvu_err_ml = np.load('../color_color/data_output/color_nuvu_err_ml.npy')
color_nuvb_err_ml = np.load('../color_color/data_output/color_nuvb_err_ml.npy')
detect_vi_ml = np.load('../color_color/data_output/detect_vi_ml.npy')
detect_ub_ml = np.load('../color_color/data_output/detect_ub_ml.npy')
detect_bv_ml = np.load('../color_color/data_output/detect_bv_ml.npy')
detect_nuvu_ml = np.load('../color_color/data_output/detect_nuvu_ml.npy')
detect_nuvb_ml = np.load('../color_color/data_output/detect_nuvb_ml.npy')
clcl_color_ml = np.load('../color_color/data_output/clcl_color_ml.npy')
clcl_qual_color_ml = np.load('../color_color/data_output/clcl_qual_color_ml.npy')
age_ml = np.load('../color_color/data_output/age_ml.npy')
ebv_ml = np.load('../color_color/data_output/ebv_ml.npy')
mag_mask_ml = np.load('../color_color/data_output/mag_mask_ml.npy')

mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_detect_ubvi_hum = detect_vi_hum * detect_ub_hum
mask_detect_ubvi_ml = detect_vi_ml * detect_ub_ml

mask_detect_bvvi_hum = detect_vi_hum * detect_bv_hum
mask_detect_bvvi_ml = detect_vi_ml * detect_bv_ml

mask_good_colors_ubvi_hum = ((color_vi_hum > -1.5) & (color_vi_hum < 2.5) &
                             (color_ub_hum > -3) & (color_ub_hum < 1.5)) * mask_detect_ubvi_hum
mask_good_colors_ubvi_ml = ((color_vi_ml > -1.5) & (color_vi_ml < 2.5) &
                            (color_ub_ml > -3) & (color_ub_ml < 1.5)) * mask_detect_ubvi_ml
mask_good_colors_bvvi_hum = ((color_vi_hum > -1.5) & (color_vi_hum < 2.5) &
                             (color_bv_hum > -1) & (color_bv_hum < 1.7)) * mask_detect_bvvi_hum
mask_good_colors_bvvi_ml = ((color_vi_ml > -1.5) & (color_vi_ml < 2.5) &
                            (color_bv_ml > -1) & (color_bv_ml < 1.7)) * mask_detect_bvvi_ml

# color range limitations
x_lim_ubvi = (-0.6, 1.9)
y_lim_ubvi = (0.9, -1.9)
n_bins_ubvi = 50


# bins
x_bins = np.linspace(x_lim_ubvi[0], x_lim_ubvi[1], n_bins_ubvi)
y_bins = np.linspace(y_lim_ubvi[1], y_lim_ubvi[0], n_bins_ubvi)
threshold_hum = 5
threshold_ml = 5

ebv_map_hum_1 = np.zeros((len(x_bins), len(y_bins))) * np.nan
ebv_map_hum_2 = np.zeros((len(x_bins), len(y_bins))) * np.nan
ebv_map_hum_3 = np.zeros((len(x_bins), len(y_bins))) * np.nan
ebv_map_ml_1 = np.zeros((len(x_bins), len(y_bins))) * np.nan
ebv_map_ml_2 = np.zeros((len(x_bins), len(y_bins))) * np.nan
ebv_map_ml_3 = np.zeros((len(x_bins), len(y_bins))) * np.nan


for x_index in range(len(x_bins)-1):
    for y_index in range(len(y_bins)-1):
        mask_in_bin_hum = ((color_vi_hum > x_bins[x_index]) & (color_vi_hum < x_bins[x_index + 1]) &
                       (color_ub_hum > y_bins[y_index]) & (color_ub_hum < y_bins[y_index + 1]))
        mask_in_bin_ml = ((color_vi_ml > x_bins[x_index]) & (color_vi_ml < x_bins[x_index + 1]) &
                       (color_ub_ml > y_bins[y_index]) & (color_ub_ml < y_bins[y_index + 1]))

        mask_selected_obj_hum_1 = mask_in_bin_hum * mask_class_1_hum * mask_good_colors_ubvi_hum
        mask_selected_obj_hum_2 = mask_in_bin_hum * mask_class_2_hum * mask_good_colors_ubvi_hum
        mask_selected_obj_hum_3 = mask_in_bin_hum * mask_class_3_hum * mask_good_colors_ubvi_hum
        mask_selected_obj_ml_1 = mask_in_bin_ml * mask_class_1_ml * mask_good_colors_ubvi_ml
        mask_selected_obj_ml_2 = mask_in_bin_ml * mask_class_2_ml * mask_good_colors_ubvi_ml
        mask_selected_obj_ml_3 = mask_in_bin_ml * mask_class_3_ml * mask_good_colors_ubvi_ml

        if sum(mask_selected_obj_hum_1) > threshold_hum:
            ebv_map_hum_1[x_index, y_index] = np.nanmean(ebv_hum[mask_selected_obj_hum_1])
        if sum(mask_selected_obj_hum_2) > threshold_hum:
            ebv_map_hum_2[x_index, y_index] = np.nanmean(ebv_hum[mask_selected_obj_hum_2])
        if sum(mask_selected_obj_hum_3) > threshold_hum:
            ebv_map_hum_3[x_index, y_index] = np.nanmean(ebv_hum[mask_selected_obj_hum_3])

        if sum(mask_selected_obj_ml_1) > threshold_ml:
            ebv_map_ml_1[x_index, y_index] = np.nanmean(ebv_ml[mask_selected_obj_ml_1])
        if sum(mask_selected_obj_ml_2) > threshold_ml:
            ebv_map_ml_2[x_index, y_index] = np.nanmean(ebv_ml[mask_selected_obj_ml_2])
        if sum(mask_selected_obj_ml_3) > threshold_ml:
            ebv_map_ml_3[x_index, y_index] = np.nanmean(ebv_ml[mask_selected_obj_ml_3])



figure = plt.figure(figsize=(29, 20))
fontsize = 28


cmap_ebv = matplotlib.cm.get_cmap('rainbow')
norm_ebv = matplotlib.colors.Normalize(vmin=0, vmax=0.5)



ax_ebv_hum_1 = figure.add_axes([0.04, 0.52, 0.32, 0.46])
ax_ebv_ml_1 = figure.add_axes([0.04, 0.05, 0.32, 0.46])

ax_ebv_hum_2 = figure.add_axes([0.33, 0.52, 0.32, 0.46])
ax_ebv_ml_2 = figure.add_axes([0.33, 0.05, 0.32, 0.46])

ax_ebv_hum_3 = figure.add_axes([0.62, 0.52, 0.32, 0.46])
ax_ebv_ml_3 = figure.add_axes([0.62, 0.05, 0.32, 0.46])

ax_cbar_ebv = figure.add_axes([0.94, 0.2, 0.015, 0.6])


ax_ebv_hum_1.imshow(ebv_map_hum_1.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_hum_2.imshow(ebv_map_hum_2.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_hum_3.imshow(ebv_map_hum_3.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_ebv, norm=norm_ebv)

ax_ebv_ml_1.imshow(ebv_map_ml_1.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_ml_2.imshow(ebv_map_ml_2.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_ml_3.imshow(ebv_map_ml_3.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_ebv, norm=norm_ebv)




ax_ebv_hum_1.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-', label=r'BC03, Z$_{\odot}$')
ax_ebv_hum_1.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}/50$')
ax_ebv_hum_2.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_ebv_hum_2.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

ax_ebv_hum_3.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_ebv_hum_3.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')
ax_ebv_ml_1.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_ebv_ml_1.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

ax_ebv_ml_2.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_ebv_ml_2.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')
ax_ebv_ml_3.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_ebv_ml_3.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

helper_func.plot_reddening_vect(ax=ax_ebv_hum_1, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
helper_func.plot_reddening_vect(ax=ax_ebv_hum_2, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
helper_func.plot_reddening_vect(ax=ax_ebv_hum_3, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
helper_func.plot_reddening_vect(ax=ax_ebv_ml_1, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
helper_func.plot_reddening_vect(ax=ax_ebv_ml_2, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
helper_func.plot_reddening_vect(ax=ax_ebv_ml_3, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)

ColorbarBase(ax_cbar_ebv, orientation='vertical', cmap=cmap_ebv, norm=norm_ebv, extend='neither', ticks=None)
ax_cbar_ebv.set_ylabel(r'E(B-V)', labelpad=-4, fontsize=fontsize + 5)
ax_cbar_ebv.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)



ax_ebv_hum_1.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                 'Class 1 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_ebv_ml_1.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                 'Class 2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_ebv_hum_2.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                 'Compact associations (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_ebv_ml_2.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                 'Class 1 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_ebv_hum_3.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                 'Class 2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_ebv_ml_3.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                 'Compact associations (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

ax_ebv_hum_1.set_xlim(x_lim_ubvi)
ax_ebv_hum_2.set_xlim(x_lim_ubvi)
ax_ebv_hum_3.set_xlim(x_lim_ubvi)
ax_ebv_ml_1.set_xlim(x_lim_ubvi)
ax_ebv_ml_2.set_xlim(x_lim_ubvi)
ax_ebv_ml_3.set_xlim(x_lim_ubvi)

ax_ebv_hum_1.set_ylim(y_lim_ubvi)
ax_ebv_hum_2.set_ylim(y_lim_ubvi)
ax_ebv_hum_3.set_ylim(y_lim_ubvi)
ax_ebv_ml_1.set_ylim(y_lim_ubvi)
ax_ebv_ml_2.set_ylim(y_lim_ubvi)
ax_ebv_ml_3.set_ylim(y_lim_ubvi)

ax_ebv_hum_2.set_yticklabels([])
ax_ebv_hum_3.set_yticklabels([])
ax_ebv_ml_2.set_yticklabels([])
ax_ebv_ml_3.set_yticklabels([])

ax_ebv_hum_1.set_xticklabels([])
ax_ebv_hum_2.set_xticklabels([])
ax_ebv_hum_3.set_xticklabels([])


ax_ebv_hum_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_ebv_ml_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_ebv_ml_1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_ebv_ml_2.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_ebv_ml_3.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_ebv_hum_1.legend(frameon=False, loc=3, fontsize=fontsize)

ax_ebv_hum_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_hum_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_hum_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_ml_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_ml_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_ml_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



plt.tight_layout()
plt.savefig('plot_output/color_color_ebv.png')
plt.savefig('plot_output/color_color_ebv.pdf')


