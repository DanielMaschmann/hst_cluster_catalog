import numpy as np
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from matplotlib import patheffects
from matplotlib.colorbar import ColorbarBase

nuvb_label_dict = {
    1: {'offsets': [0.25, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}
ub_label_dict = {
    1: {'offsets': [0.2, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}
bv_label_dict = {
    1: {'offsets': [0.2, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.1], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.1], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}

nuvb_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.7, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [+0.05, -0.9], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
ub_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
bv_annotation_dict = {
    500: {'offset': [-0.5, +0.3], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}


def display_models(ax, y_color='nuvb',
                   age_cut_sol50=5e2,
                   age_dots_sol=None,
                   age_dots_sol50=None,
                   age_labels=False,
                   age_label_color='red',
                   age_label_fontsize=30,
                   color_sol='tab:cyan', linewidth_sol=4, linestyle_sol='-',
                   color_sol50='m', linewidth_sol50=4, linestyle_sol50='-',
                   label_sol=None, label_sol50=None):

    y_model_sol = globals()['model_%s_sol' % y_color]
    y_model_sol50 = globals()['model_%s_sol50' % y_color]

    ax.plot(model_vi_sol, y_model_sol, color=color_sol, linewidth=linewidth_sol, linestyle=linestyle_sol, zorder=10,
            label=label_sol)
    ax.plot(model_vi_sol50[age_mod_sol50 > age_cut_sol50], y_model_sol50[age_mod_sol50 > age_cut_sol50],
            color=color_sol50, linewidth=linewidth_sol50, linestyle=linestyle_sol50, zorder=10, label=label_sol50)

    if age_dots_sol is None:
        age_dots_sol = [1, 5, 10, 100, 500, 1000, 13750]
    for age in age_dots_sol:
        ax.scatter(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age], color='b', s=80, zorder=20)

    if age_dots_sol50 is None:
        age_dots_sol50 = [500, 1000, 13750]
    for age in age_dots_sol50:
        ax.scatter(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age], color='tab:pink', s=80, zorder=20)

    if age_labels:
        label_dict = globals()['%s_label_dict' % y_color]
        pe = [patheffects.withStroke(linewidth=3, foreground="w")]
        for age in label_dict.keys():

            ax.text(model_vi_sol[age_mod_sol == age]+label_dict[age]['offsets'][0],
                    y_model_sol[age_mod_sol == age]+label_dict[age]['offsets'][1],
                    label_dict[age]['label'], horizontalalignment=label_dict[age]['ha'], verticalalignment=label_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize,
                    path_effects=pe)

        annotation_dict = globals()['%s_annotation_dict' % y_color]
        for age in annotation_dict.keys():

            txt_sol = ax.annotate(' ', #annotation_dict[age]['label'],
                        xy=(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data', color=age_label_color,
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkcyan', lw=3, ls='-'),
                        path_effects=[patheffects.withStroke(linewidth=3,
                                                        foreground="w")])
            txt_sol.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
                                                  patheffects.Normal()])
            txt_sol50 = ax.annotate(' ',
                        xy=(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data',
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkviolet', lw=3, ls='-'),
                        path_effects=[patheffects.withStroke(linewidth=3,
                                                        foreground="w")])
            txt_sol50.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
                                                  patheffects.Normal()])
            ax.text(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                    y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1],
                    annotation_dict[age]['label'],
                    horizontalalignment=annotation_dict[age]['ha'], verticalalignment=annotation_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize, zorder=40, path_effects=pe)


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
mask_class_12_hum = mask_class_1_hum + mask_class_2_hum

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3
mask_class_12_ml = mask_class_1_ml + mask_class_2_ml

mask_detect_nuvbvi_hum = detect_nuv_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum

mask_detect_nuvbvi_ml = detect_nuv_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml


x_lim_vi = (-0.7, 2.4)
y_lim_nuvb = (3.2, -2.9)
y_lim_ub = (2.1, -2.2)


mask_good_colors_nuvbvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_nuvb_hum > (y_lim_nuvb[1] - 1)) & (color_nuvb_hum < (y_lim_nuvb[0] + 1)) &
                               mask_detect_nuvbvi_hum)
mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_hum)
mask_good_colors_nuvbvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_nuvb_ml > (y_lim_nuvb[1] - 1)) & (color_nuvb_ml < (y_lim_nuvb[0] + 1)) &
                               mask_detect_nuvbvi_ml)
mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_ml)

# bins
n_bins_uncertainty = 50
x_bins_vi = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_uncertainty)
y_bins_nuvb = np.linspace(y_lim_nuvb[1], y_lim_nuvb[0], n_bins_uncertainty)
y_bins_ub = np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins_uncertainty)
threshold_hum = 5
threshold_ml = 5


snr_vi_map_hum_nuvbvi = np.zeros((len(x_bins_vi), len(y_bins_nuvb))) * np.nan
snr_vi_map_ml_nuvbvi = np.zeros((len(x_bins_vi), len(y_bins_nuvb))) * np.nan
snr_nuvb_map_hum_nuvbvi = np.zeros((len(x_bins_vi), len(y_bins_nuvb))) * np.nan
snr_nuvb_map_ml_nuvbvi = np.zeros((len(x_bins_vi), len(y_bins_nuvb))) * np.nan

snr_vi_map_hum_ubvi = np.zeros((len(x_bins_vi), len(y_bins_ub))) * np.nan
snr_vi_map_ml_ubvi = np.zeros((len(x_bins_vi), len(y_bins_ub))) * np.nan
snr_ub_map_hum_ubvi = np.zeros((len(x_bins_vi), len(y_bins_ub))) * np.nan
snr_ub_map_ml_ubvi = np.zeros((len(x_bins_vi), len(y_bins_ub))) * np.nan

for x_index in range(len(x_bins_vi)-1):
    for y_index in range(len(y_bins_nuvb)-1):
        mask_in_bin_hum_nuvbvi = ((color_vi_hum > x_bins_vi[x_index]) & (color_vi_hum < x_bins_vi[x_index + 1]) &
                                  (color_nuvb_hum > y_bins_nuvb[y_index]) & (color_nuvb_hum < y_bins_nuvb[y_index + 1]))
        mask_in_bin_ml_nuvbvi = ((color_vi_ml > x_bins_vi[x_index]) & (color_vi_ml < x_bins_vi[x_index + 1]) &
                                 (color_nuvb_ml > y_bins_nuvb[y_index]) & (color_nuvb_ml < y_bins_nuvb[y_index + 1]))
        mask_in_bin_hum_ubvi = ((color_vi_hum > x_bins_vi[x_index]) & (color_vi_hum < x_bins_vi[x_index + 1]) &
                                  (color_ub_hum > y_bins_ub[y_index]) & (color_ub_hum < y_bins_ub[y_index + 1]))
        mask_in_bin_ml_ubvi = ((color_vi_ml > x_bins_vi[x_index]) & (color_vi_ml < x_bins_vi[x_index + 1]) &
                                 (color_ub_ml > y_bins_ub[y_index]) & (color_ub_ml < y_bins_ub[y_index + 1]))

        mask_selected_obj_hum_nuvbvi = mask_in_bin_hum_nuvbvi * mask_class_12_hum * mask_good_colors_nuvbvi_hum
        mask_selected_obj_ml_nuvbvi = mask_in_bin_ml_nuvbvi * mask_class_12_ml * mask_good_colors_nuvbvi_ml
        mask_selected_obj_hum_ubvi = mask_in_bin_hum_ubvi * mask_class_12_hum * mask_good_colors_ubvi_hum
        mask_selected_obj_ml_ubvi = mask_in_bin_ml_ubvi * mask_class_12_ml * mask_good_colors_ubvi_ml

        if sum(mask_selected_obj_hum_nuvbvi) > threshold_hum:
            snr_vi_map_hum_nuvbvi[x_index, y_index] = np.nanmean(color_vi_err_hum[mask_selected_obj_hum_nuvbvi])
            snr_nuvb_map_hum_nuvbvi[x_index, y_index] = np.nanmean(color_nuvb_err_hum[mask_selected_obj_hum_nuvbvi])
        if sum(mask_selected_obj_ml_nuvbvi) > threshold_ml:
            snr_vi_map_ml_nuvbvi[x_index, y_index] = np.nanmean(color_vi_err_ml[mask_selected_obj_ml_nuvbvi])
            snr_nuvb_map_ml_nuvbvi[x_index, y_index] = np.nanmean(color_nuvb_err_ml[mask_selected_obj_ml_nuvbvi])

        if sum(mask_selected_obj_hum_ubvi) > threshold_hum:
            snr_vi_map_hum_ubvi[x_index, y_index] = np.nanmean(color_vi_err_hum[mask_selected_obj_hum_ubvi])
            snr_ub_map_hum_ubvi[x_index, y_index] = np.nanmean(color_ub_err_hum[mask_selected_obj_hum_ubvi])
        if sum(mask_selected_obj_ml_ubvi) > threshold_ml:
            snr_vi_map_ml_ubvi[x_index, y_index] = np.nanmean(color_vi_err_ml[mask_selected_obj_ml_ubvi])
            snr_ub_map_ml_ubvi[x_index, y_index] = np.nanmean(color_ub_err_ml[mask_selected_obj_ml_ubvi])


figure = plt.figure(figsize=(30, 20))
fontsize = 27

cmap_vi = matplotlib.cm.get_cmap('Purples')
norm_vi = matplotlib.colors.Normalize(vmin=0, vmax=0.11)

# cmap_vi_ub = matplotlib.cm.get_cmap('plasma')
# norm_vi_ub = matplotlib.colors.Normalize(vmin=0, vmax=0.11)

cmap_nuvb_nuvb = matplotlib.cm.get_cmap('inferno')
norm_nuvb_nuvb = matplotlib.colors.Normalize(vmin=0, vmax=0.8)

cmap_ub_ub = matplotlib.cm.get_cmap('cividis')
norm_ub_ub = matplotlib.colors.Normalize(vmin=0, vmax=0.5)


ax_cc_sn_vi_hum_nuvb = figure.add_axes([0.04, 0.53, 0.235, 0.4])
ax_cc_sn_vi_ml_nuvb = figure.add_axes([0.28, 0.53, 0.235, 0.4])

ax_cc_sn_nuvb_hum_nuvb = figure.add_axes([0.52, 0.53, 0.235, 0.4])
ax_cc_sn_nuvb_ml_nuvb = figure.add_axes([0.76, 0.53, 0.235, 0.4])

ax_cc_sn_vi_hum_ub = figure.add_axes([0.04, 0.05, 0.235, 0.4])
ax_cc_sn_vi_ml_ub = figure.add_axes([0.28, 0.05, 0.235, 0.4])

ax_cc_sn_ub_hum_ub = figure.add_axes([0.52, 0.05, 0.235, 0.4])
ax_cc_sn_ub_ml_ub = figure.add_axes([0.76, 0.05, 0.235, 0.4])


ax_cbar_sn_vi = figure.add_axes([0.12, 0.94, 0.3, 0.015])
ax_cbar_sn_nuvb_nuvb = figure.add_axes([0.6, 0.94, 0.3, 0.015])

# ax_cbar_sn_vi_ub = figure.add_axes([0.12, 0.46, 0.3, 0.015])
ax_cbar_sn_ub_ub = figure.add_axes([0.6, 0.46, 0.3, 0.015])


ax_cc_sn_vi_hum_nuvb.imshow(snr_vi_map_hum_nuvbvi.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_nuvb[1], y_lim_nuvb[0]),
                       cmap=cmap_vi, norm=norm_vi,  interpolation='nearest', aspect='auto')
ax_cc_sn_vi_ml_nuvb.imshow(snr_vi_map_ml_nuvbvi.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_nuvb[1], y_lim_nuvb[0]),
                      cmap=cmap_vi, norm=norm_vi,  interpolation='nearest', aspect='auto')

ax_cc_sn_nuvb_hum_nuvb.imshow(snr_nuvb_map_hum_nuvbvi.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_nuvb[1], y_lim_nuvb[0]),
                       cmap=cmap_nuvb_nuvb, norm=norm_nuvb_nuvb,  interpolation='nearest', aspect='auto')
ax_cc_sn_nuvb_ml_nuvb.imshow(snr_nuvb_map_ml_nuvbvi.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_nuvb[1], y_lim_nuvb[0]),
                      cmap=cmap_nuvb_nuvb, norm=norm_nuvb_nuvb,  interpolation='nearest', aspect='auto')

ax_cc_sn_vi_hum_ub.imshow(snr_vi_map_hum_ubvi.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                       cmap=cmap_vi, norm=norm_vi,  interpolation='nearest', aspect='auto')
ax_cc_sn_vi_ml_ub.imshow(snr_vi_map_ml_ubvi.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                      cmap=cmap_vi, norm=norm_vi,  interpolation='nearest', aspect='auto')

ax_cc_sn_ub_hum_ub.imshow(snr_ub_map_hum_ubvi.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                       cmap=cmap_ub_ub, norm=norm_ub_ub,  interpolation='nearest', aspect='auto')
ax_cc_sn_ub_ml_ub.imshow(snr_ub_map_ml_ubvi.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                      cmap=cmap_ub_ub, norm=norm_ub_ub,  interpolation='nearest', aspect='auto')


ColorbarBase(ax_cbar_sn_vi, orientation='horizontal', cmap=cmap_vi, norm=norm_vi, extend='neither', ticks=None)
ax_cbar_sn_vi.set_xlabel(r'$\sigma_{\rm V-I}$', labelpad=10, fontsize=fontsize)
ax_cbar_sn_vi.tick_params(axis='both', which='both', width=2, direction='in',
                               top=True, labeltop=True,  bottom=False, labelbottom=False, labelsize=fontsize)
ax_cbar_sn_vi.xaxis.set_label_position('top')

ColorbarBase(ax_cbar_sn_nuvb_nuvb, orientation='horizontal', cmap=cmap_nuvb_nuvb, norm=norm_nuvb_nuvb, extend='neither', ticks=None)
ax_cbar_sn_nuvb_nuvb.set_xlabel(r'$\sigma_{\rm NUV-B}$', labelpad=10, fontsize=fontsize + 5)
ax_cbar_sn_nuvb_nuvb.tick_params(axis='both', which='both', width=2, direction='in',
                               top=True, labeltop=True,  bottom=False, labelbottom=False, labelsize=fontsize)
ax_cbar_sn_nuvb_nuvb.xaxis.set_label_position('top')


ColorbarBase(ax_cbar_sn_vi, orientation='horizontal', cmap=cmap_vi, norm=norm_vi, extend='neither', ticks=None)
ax_cbar_sn_vi.set_xlabel(r'$\sigma_{\rm V-I}$', labelpad=10, fontsize=fontsize)
ax_cbar_sn_vi.tick_params(axis='both', which='both', width=2, direction='in',
                               top=True, labeltop=True,  bottom=False, labelbottom=False, labelsize=fontsize)
ax_cbar_sn_vi.xaxis.set_label_position('top')

ColorbarBase(ax_cbar_sn_ub_ub, orientation='horizontal', cmap=cmap_ub_ub, norm=norm_ub_ub, extend='neither', ticks=None)
ax_cbar_sn_ub_ub.set_xlabel(r'$\sigma_{\rm U-B}$', labelpad=10, fontsize=fontsize + 5)
ax_cbar_sn_ub_ub.tick_params(axis='both', which='both', width=2, direction='in',
                               top=True, labeltop=True,  bottom=False, labelbottom=False, labelsize=fontsize)
ax_cbar_sn_ub_ub.xaxis.set_label_position('top')


display_models(ax=ax_cc_sn_vi_hum_nuvb, age_label_fontsize=fontsize+2, label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax_cc_sn_vi_ml_nuvb, age_label_fontsize=fontsize+2)
display_models(ax=ax_cc_sn_nuvb_hum_nuvb, age_label_fontsize=fontsize+2)
display_models(ax=ax_cc_sn_nuvb_ml_nuvb, age_label_fontsize=fontsize+2)

display_models(ax=ax_cc_sn_vi_hum_ub, age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax_cc_sn_vi_ml_ub, age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax_cc_sn_ub_hum_ub, age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax_cc_sn_ub_ml_ub, age_label_fontsize=fontsize+2, y_color='ub')


ax_cc_sn_vi_hum_nuvb.text(0.95, 0.95, 'Class 1 + 2 (Hum)', horizontalalignment='right', verticalalignment='center',
                          fontsize=fontsize, transform=ax_cc_sn_vi_hum_nuvb.transAxes)
ax_cc_sn_vi_ml_nuvb.text(0.95, 0.95, 'Class 1 + 2 (ML)', horizontalalignment='right', verticalalignment='center',
                          fontsize=fontsize, transform=ax_cc_sn_vi_ml_nuvb.transAxes)
ax_cc_sn_nuvb_hum_nuvb.text(0.95, 0.95, 'Class 1 + 2 (Hum)', horizontalalignment='right', verticalalignment='center',
                          fontsize=fontsize, transform=ax_cc_sn_nuvb_hum_nuvb.transAxes)
ax_cc_sn_nuvb_ml_nuvb.text(0.95, 0.95, 'Class 1 + 2 (ML)', horizontalalignment='right', verticalalignment='center',
                          fontsize=fontsize, transform=ax_cc_sn_nuvb_ml_nuvb.transAxes)

ax_cc_sn_vi_hum_ub.text(0.95, 0.95, 'Class 1 + 2 (Hum)', horizontalalignment='right', verticalalignment='center',
                          fontsize=fontsize, transform=ax_cc_sn_vi_hum_ub.transAxes)
ax_cc_sn_vi_ml_ub.text(0.95, 0.95, 'Class 1 + 2 (ML)', horizontalalignment='right', verticalalignment='center',
                          fontsize=fontsize, transform=ax_cc_sn_vi_ml_ub.transAxes)
ax_cc_sn_ub_hum_ub.text(0.95, 0.95, 'Class 1 + 2 (Hum)', horizontalalignment='right', verticalalignment='center',
                          fontsize=fontsize, transform=ax_cc_sn_ub_hum_ub.transAxes)
ax_cc_sn_ub_ml_ub.text(0.95, 0.95, 'Class 1 + 2 (ML)', horizontalalignment='right', verticalalignment='center',
                          fontsize=fontsize, transform=ax_cc_sn_ub_ml_ub.transAxes)

ax_cc_sn_vi_hum_nuvb.set_xlim(x_lim_vi)
ax_cc_sn_vi_ml_nuvb.set_xlim(x_lim_vi)
ax_cc_sn_nuvb_hum_nuvb.set_xlim(x_lim_vi)
ax_cc_sn_nuvb_ml_nuvb.set_xlim(x_lim_vi)
ax_cc_sn_vi_hum_ub.set_xlim(x_lim_vi)
ax_cc_sn_vi_ml_ub.set_xlim(x_lim_vi)
ax_cc_sn_ub_hum_ub.set_xlim(x_lim_vi)
ax_cc_sn_ub_ml_ub.set_xlim(x_lim_vi)

ax_cc_sn_vi_hum_nuvb.set_ylim(y_lim_nuvb)
ax_cc_sn_vi_ml_nuvb.set_ylim(y_lim_nuvb)
ax_cc_sn_nuvb_hum_nuvb.set_ylim(y_lim_nuvb)
ax_cc_sn_nuvb_ml_nuvb.set_ylim(y_lim_nuvb)

ax_cc_sn_vi_hum_ub.set_ylim(y_lim_ub)
ax_cc_sn_vi_ml_ub.set_ylim(y_lim_ub)
ax_cc_sn_ub_hum_ub.set_ylim(y_lim_ub)
ax_cc_sn_ub_ml_ub.set_ylim(y_lim_ub)



ax_cc_sn_vi_ml_nuvb.set_yticklabels([])
ax_cc_sn_nuvb_hum_nuvb.set_yticklabels([])
ax_cc_sn_nuvb_ml_nuvb.set_yticklabels([])

ax_cc_sn_vi_hum_ub.set_yticks([-2, 1, 0, 1, 2])
ax_cc_sn_vi_ml_ub.set_yticks([-2, 1, 0, 1, 2])
ax_cc_sn_ub_hum_ub.set_yticks([-2, 1, 0, 1, 2])
ax_cc_sn_ub_ml_ub.set_yticks([-2, 1, 0, 1, 2])

ax_cc_sn_vi_ml_ub.set_yticklabels([])
ax_cc_sn_ub_hum_ub.set_yticklabels([])
ax_cc_sn_ub_ml_ub.set_yticklabels([])

ax_cc_sn_vi_hum_nuvb.set_xticklabels([])
ax_cc_sn_vi_ml_nuvb.set_xticklabels([])
ax_cc_sn_nuvb_hum_nuvb.set_xticklabels([])
ax_cc_sn_nuvb_ml_nuvb.set_xticklabels([])

ax_cc_sn_vi_hum_ub.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_cc_sn_vi_hum_nuvb.set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_cc_sn_vi_hum_ub.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_sn_vi_ml_ub.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_sn_ub_hum_ub.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_sn_ub_ml_ub.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_cc_sn_vi_hum_nuvb.legend(frameon=False, loc=3, fontsize=fontsize)

ax_cc_sn_vi_hum_nuvb.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_vi_ml_nuvb.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_nuvb_hum_nuvb.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_nuvb_ml_nuvb.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_vi_hum_ub.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_vi_ml_ub.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_ub_hum_ub.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_ub_ml_ub.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


plt.savefig('plot_output/uncert_reg_c12.png')
plt.savefig('plot_output/uncert_reg_c12.pdf')
