import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colorbar import ColorbarBase
from photometry_tools import helper_func

age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_nuvu_sol = np.load('../color_color/data_output/model_nuvu_sol.npy')
model_nuvb_sol = np.load('../color_color/data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')
model_bi_sol = np.load('../color_color/data_output/model_bi_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_nuvu_sol50 = np.load('../color_color/data_output/model_nuvu_sol50.npy')
model_nuvb_sol50 = np.load('../color_color/data_output/model_nuvb_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')
model_bi_sol50 = np.load('../color_color/data_output/model_bi_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')


color_vi_hum = np.load('../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../color_color/data_output/color_ub_hum.npy')
color_bv_hum = np.load('../color_color/data_output/color_bv_hum.npy')
color_bi_hum = np.load('../color_color/data_output/color_bi_hum.npy')
color_nuvu_hum = np.load('../color_color/data_output/color_nuvu_hum.npy')
color_nuvb_hum = np.load('../color_color/data_output/color_nuvb_hum.npy')
color_vi_err_hum = np.load('../color_color/data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('../color_color/data_output/color_ub_err_hum.npy')
color_bv_err_hum = np.load('../color_color/data_output/color_bv_err_hum.npy')
color_bi_err_hum = np.load('../color_color/data_output/color_bi_err_hum.npy')
color_nuvu_err_hum = np.load('../color_color/data_output/color_nuvu_err_hum.npy')
color_nuvb_err_hum = np.load('../color_color/data_output/color_nuvb_err_hum.npy')
detect_nuv_hum = np.load('../color_color/data_output/detect_nuv_hum.npy')
detect_u_hum = np.load('../color_color/data_output/detect_u_hum.npy')
detect_b_hum = np.load('../color_color/data_output/detect_b_hum.npy')
detect_v_hum = np.load('../color_color/data_output/detect_v_hum.npy')
detect_i_hum = np.load('../color_color/data_output/detect_i_hum.npy')
clcl_color_hum = np.load('../color_color/data_output/clcl_color_hum.npy')
age_hum = np.load('../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../color_color/data_output/ebv_hum.npy')
color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
color_bv_ml = np.load('../color_color/data_output/color_bv_ml.npy')
color_bi_ml = np.load('../color_color/data_output/color_bi_ml.npy')
color_nuvu_ml = np.load('../color_color/data_output/color_nuvu_ml.npy')
color_nuvb_ml = np.load('../color_color/data_output/color_nuvb_ml.npy')
color_vi_err_ml = np.load('../color_color/data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('../color_color/data_output/color_ub_err_ml.npy')
color_bv_err_ml = np.load('../color_color/data_output/color_bv_err_ml.npy')
color_bi_err_ml = np.load('../color_color/data_output/color_bi_err_ml.npy')
color_nuvu_err_ml = np.load('../color_color/data_output/color_nuvu_err_ml.npy')
color_nuvb_err_ml = np.load('../color_color/data_output/color_nuvb_err_ml.npy')
detect_nuv_ml = np.load('../color_color/data_output/detect_nuv_ml.npy')
detect_u_ml = np.load('../color_color/data_output/detect_u_ml.npy')
detect_b_ml = np.load('../color_color/data_output/detect_b_ml.npy')
detect_v_ml = np.load('../color_color/data_output/detect_v_ml.npy')
detect_i_ml = np.load('../color_color/data_output/detect_i_ml.npy')
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


mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml


mask_good_colors_ubvi_hum = ((color_vi_hum > -1.5) & (color_vi_hum < 2.5) &
                             (color_ub_hum > -3) & (color_ub_hum < 1.5)) * mask_detect_ubvi_hum
mask_good_colors_ubvi_ml = ((color_vi_ml > -1.5) & (color_vi_ml < 2.5) &
                            (color_ub_ml > -3) & (color_ub_ml < 1.5)) * mask_detect_ubvi_ml

# color range limitations
x_lim_ubvi = (-0.6, 1.9)
y_lim_ubvi = (0.9, -1.9)
# x_lim = (-0.6, 2.1)
# y_lim = (0.9, -2.2)
n_bins_ubvi = 120
threshold_fact = 4
kernal_std = 1.0
contrast = 0.01

gauss_dict_ubvi_hum_12 = helper_func.calc_seg(x_data=color_vi_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_ubvi_hum],
                                              y_data=color_ub_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_ubvi_hum],
                                              x_data_err=color_vi_err_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_ubvi_hum],
                                              y_data_err=color_ub_err_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_ubvi_hum],
                                              x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

gauss_dict_ubvi_ml_12 = helper_func.calc_seg(x_data=color_vi_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ubvi_ml],
                                             y_data=color_ub_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ubvi_ml],
                                             x_data_err=color_vi_err_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ubvi_ml],
                                             y_data_err=color_ub_err_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ubvi_ml],
                                             x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)


# bins
n_bins_uncertainty_ubvi = 50
x_bins = np.linspace(x_lim_ubvi[0], x_lim_ubvi[1], n_bins_uncertainty_ubvi)
y_bins = np.linspace(y_lim_ubvi[1], y_lim_ubvi[0], n_bins_uncertainty_ubvi)
threshold_hum = 5
threshold_ml = 5
# print(sum(mask_class_1_hum + mask_class_2_hum))
# print(sum(mask_class_1_ml + mask_class_2_ml))
# print(2500)
#
# n_hum = sum(mask_class_1_hum + mask_class_2_hum)
# n_ml = sum(mask_class_1_ml + mask_class_2_ml)
#
# print(n_hum / (2500 * 2))
# print(n_ml / (2500 * 2))



snr_vi_map_hum = np.zeros((len(x_bins), len(y_bins))) * np.nan
snr_vi_map_ml = np.zeros((len(x_bins), len(y_bins))) * np.nan
snr_ub_map_hum = np.zeros((len(x_bins), len(y_bins))) * np.nan
snr_ub_map_ml = np.zeros((len(x_bins), len(y_bins))) * np.nan


for x_index in range(len(x_bins)-1):
    for y_index in range(len(y_bins)-1):
        mask_in_bin_hum = ((color_vi_hum > x_bins[x_index]) & (color_vi_hum < x_bins[x_index + 1]) &
                       (color_ub_hum > y_bins[y_index]) & (color_ub_hum < y_bins[y_index + 1]))
        mask_in_bin_ml = ((color_vi_ml > x_bins[x_index]) & (color_vi_ml < x_bins[x_index + 1]) &
                       (color_ub_ml > y_bins[y_index]) & (color_ub_ml < y_bins[y_index + 1]))

        mask_selected_obj_hum = mask_in_bin_hum * (mask_class_1_hum + mask_class_2_hum) * mask_good_colors_ubvi_hum
        mask_selected_obj_ml = mask_in_bin_ml * (mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ubvi_ml

        if sum(mask_selected_obj_hum) > threshold_hum:
            snr_vi_map_hum[x_index, y_index] = np.nanmean(color_vi_err_hum[mask_selected_obj_hum])
            snr_ub_map_hum[x_index, y_index] = np.nanmean(color_ub_err_hum[mask_selected_obj_hum])
        if sum(mask_selected_obj_ml) > threshold_ml:
            snr_vi_map_ml[x_index, y_index] = np.nanmean(color_vi_err_ml[mask_selected_obj_ml])
            snr_ub_map_ml[x_index, y_index] = np.nanmean(color_ub_err_ml[mask_selected_obj_ml])



figure = plt.figure(figsize=(30, 20))
fontsize = 28


cmap_vi = matplotlib.cm.get_cmap('viridis')
norm_vi = matplotlib.colors.Normalize(vmin=0, vmax=0.11)

cmap_ub = matplotlib.cm.get_cmap('cividis')
norm_ub = matplotlib.colors.Normalize(vmin=0, vmax=0.5)


ax_cc_sn_vi_hum = figure.add_axes([0.04, 0.51, 0.30, 0.45])
ax_cc_sn_vi_ml = figure.add_axes([0.04, 0.05, 0.30, 0.45])

ax_cc_sn_ub_hum = figure.add_axes([0.38, 0.51, 0.30, 0.45])
ax_cc_sn_ub_ml = figure.add_axes([0.38, 0.05, 0.30, 0.45])

ax_cc_reg_hum = figure.add_axes([0.71, 0.51, 0.30, 0.45])
ax_cc_reg_ml = figure.add_axes([0.71, 0.05, 0.30, 0.45])

ax_cbar_sn_vi = figure.add_axes([0.33, 0.2, 0.015, 0.6])
ax_cbar_sn_ub = figure.add_axes([0.67, 0.2, 0.015, 0.6])


helper_func.plot_reg_map(ax=ax_cc_reg_hum, gauss_map=gauss_dict_ubvi_hum_12['gauss_map'], seg_map=gauss_dict_ubvi_hum_12['seg_deb_map'], smooth_kernel=kernal_std,
                         x_label='vi', y_label='ub', plot_cont_1=True, plot_cont_2=True, plot_cont_3=True, save_str_1='ubvi_12_young_hum', save_str_2='ubvi_12_cascade_hum', save_str_3='ubvi_12_gc_hum',
             x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi)

helper_func.plot_reg_map(ax=ax_cc_reg_ml, gauss_map=gauss_dict_ubvi_ml_12['gauss_map'], seg_map=gauss_dict_ubvi_ml_12['seg_deb_map'], smooth_kernel=kernal_std,
                         x_label='vi', y_label='ub', plot_cont_1=True, plot_cont_2=True, plot_cont_3=True, save_str_1='ubvi_12_young_ml', save_str_2='ubvi_12_cascade_ml', save_str_3='ubvi_12_gc_ml',

             x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi)


ax_cc_sn_vi_hum.imshow(snr_vi_map_hum.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_vi, norm=norm_vi)
ax_cc_sn_vi_ml.imshow(snr_vi_map_ml.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_vi, norm=norm_vi)
ax_cc_sn_ub_hum.imshow(snr_ub_map_hum.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_ub, norm=norm_ub)
ax_cc_sn_ub_ml.imshow(snr_ub_map_ml.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_ub, norm=norm_ub)

ax_cc_sn_vi_hum.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-', label=r'BC03, Z$_{\odot}$')
ax_cc_sn_vi_hum.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}/50$')
ax_cc_sn_vi_ml.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_cc_sn_vi_ml.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

ax_cc_sn_ub_hum.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_cc_sn_ub_hum.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')
ax_cc_sn_ub_ml.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_cc_sn_ub_ml.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

ax_cc_reg_hum.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_cc_reg_hum.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')
ax_cc_reg_ml.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_cc_reg_ml.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

# helper_func.plot_reddening_vect(ax=ax_cc_sn_vi_hum, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
# helper_func.plot_reddening_vect(ax=ax_cc_sn_vi_ml, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
# helper_func.plot_reddening_vect(ax=ax_cc_sn_ub_hum, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
# helper_func.plot_reddening_vect(ax=ax_cc_sn_ub_ml, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
# helper_func.plot_reddening_vect(ax=ax_cc_reg_hum, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)
# helper_func.plot_reddening_vect(ax=ax_cc_reg_ml, vi_int=1.1, ub_int=-1.4, max_av=1, fontsize=fontsize, linewidth=3)

vi_int = 1.0
ub_int = -1.3
av_value = 1

helper_func.plot_reddening_vect(ax=ax_cc_sn_vi_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='tab:red', text=True, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_sn_vi_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='tab:red', text=True, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_sn_ub_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='tab:red', text=True, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_sn_ub_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='tab:red', text=True, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_reg_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='tab:red', text=True, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_reg_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='tab:red', text=True, fontsize=fontsize)






ColorbarBase(ax_cbar_sn_vi, orientation='vertical', cmap=cmap_vi, norm=norm_vi, extend='neither', ticks=None)
ax_cbar_sn_vi.set_ylabel(r'$\sigma_{\rm V-I}$', labelpad=-4, fontsize=fontsize + 5)
ax_cbar_sn_vi.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)

ColorbarBase(ax_cbar_sn_ub, orientation='vertical', cmap=cmap_ub, norm=norm_ub, extend='neither', ticks=None)
ax_cbar_sn_ub.set_ylabel(r'$\sigma_{\rm U-B}$', labelpad=-4, fontsize=fontsize + 5)
ax_cbar_sn_ub.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)


ax_cc_sn_vi_hum.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_sn_vi_ml.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_sn_ub_hum.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_sn_ub_ml.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_hum.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_ml.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

ax_cc_sn_vi_hum.set_title('Mean V-I uncertainty map', fontsize=fontsize)
ax_cc_sn_ub_hum.set_title('Mean U-B uncertainty map', fontsize=fontsize)
ax_cc_reg_hum.set_title('Uncertainty weighted density', fontsize=fontsize)

ax_cc_sn_vi_hum.set_xlim(x_lim_ubvi)
ax_cc_sn_vi_ml.set_xlim(x_lim_ubvi)
ax_cc_sn_ub_hum.set_xlim(x_lim_ubvi)
ax_cc_sn_ub_ml.set_xlim(x_lim_ubvi)
ax_cc_reg_hum.set_xlim(x_lim_ubvi)
ax_cc_reg_ml.set_xlim(x_lim_ubvi)

ax_cc_sn_vi_hum.set_ylim(y_lim_ubvi)
ax_cc_sn_vi_ml.set_ylim(y_lim_ubvi)
ax_cc_sn_ub_hum.set_ylim(y_lim_ubvi)
ax_cc_sn_ub_ml.set_ylim(y_lim_ubvi)
ax_cc_reg_hum.set_ylim(y_lim_ubvi)
ax_cc_reg_ml.set_ylim(y_lim_ubvi)

ax_cc_reg_hum.set_yticklabels([])
ax_cc_reg_ml.set_yticklabels([])
ax_cc_sn_ub_hum.set_yticklabels([])
ax_cc_sn_ub_ml.set_yticklabels([])

ax_cc_sn_vi_hum.set_xticklabels([])
ax_cc_sn_ub_hum.set_xticklabels([])
ax_cc_reg_hum.set_xticklabels([])

# ax_cc_reg_hum.set_title('Human Class 1|2', fontsize=fontsize)
# ax_cc_reg_ml.set_title('ML Class 1|2', fontsize=fontsize)
ax_cc_sn_vi_hum.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_cc_sn_vi_ml.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_cc_sn_vi_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_sn_ub_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_reg_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_cc_sn_vi_hum.legend(frameon=False, loc=3, fontsize=fontsize)

ax_cc_sn_vi_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_vi_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_ub_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_ub_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_reg_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_reg_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


# plt.show()
# plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('plot_output/uncert_reg_c12.png')
plt.savefig('plot_output/uncert_reg_c12.pdf')







#
#
#
#
# def calc_seg(x_data, y_data, x_data_err, y_data_err, x_lim, y_lim, n_bins, threshold_fact=2):
#
#     # calculate combined errors
#     data_err = np.sqrt(x_data_err**2 + y_data_err**2)
#     noise_cut = np.percentile(data_err, 90)
#
#     # bins
#     x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
#     y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
#     # get a mesh
#     x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
#     gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
#     noise_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
#
#     for color_index in range(len(x_data)):
#         gauss = gauss2d(x=x_mesh, y=y_mesh, x0=x_data[color_index], y0=y_data[color_index],
#                         sig_x=x_data_err[color_index], sig_y=y_data_err[color_index])
#         gauss_map += gauss
#         if data_err[color_index] > noise_cut:
#             noise_map += gauss
#
#     gauss_map -= np.nanmean(noise_map)
#
#     kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
#
#     conv_gauss_map = convolve(gauss_map, kernel)
#     threshold = len(x_data) / threshold_fact
#     # threshold = np.nanmax(conv_gauss_map) / threshold_fact
#
#     seg_map = detect_sources(conv_gauss_map, threshold, npixels=20)
#     seg_deb_map = deblend_sources(conv_gauss_map, seg_map, npixels=20, nlevels=32, contrast=0.001, progress_bar=False)
#
#     return_dict = {
#         'gauss_map': gauss_map, 'conv_gauss_map': conv_gauss_map, 'seg_deb_map': seg_deb_map}
#
#     return return_dict
#
#
#
# def plot_reg_map(ax, gauss_map, seg_map, x_lim, y_lim):
#
#     gauss_map_no_seg = gauss_map.copy()
#     gauss_map_seg1 = gauss_map.copy()
#     gauss_map_seg2 = gauss_map.copy()
#     gauss_map_seg3 = gauss_map.copy()
#     gauss_map_no_seg[seg_map._data != 0] = np.nan
#     gauss_map_seg1[seg_map._data != 1] = np.nan
#     gauss_map_seg2[seg_map._data != 2] = np.nan
#     gauss_map_seg3[seg_map._data != 3] = np.nan
#     ax.imshow(gauss_map_no_seg, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
#                     cmap='Greys', vmin=0, vmax=np.nanmax(gauss_map)/1.2)
#     ax.imshow(gauss_map_seg1, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
#                     cmap='Blues', vmin=0, vmax=np.nanmax(gauss_map)/1.2)
#     ax.imshow(gauss_map_seg2, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
#                     cmap='Greens', vmin=0, vmax=np.nanmax(gauss_map)/0.9)
#     ax.imshow(gauss_map_seg3, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
#                     cmap='Reds', vmin=0, vmax=np.nanmax(gauss_map)/1.4)
#
#
#
#     ax.set_xlim(x_lim)
#     ax.set_ylim(y_lim)
#
