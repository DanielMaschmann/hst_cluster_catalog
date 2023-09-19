import numpy as np
import matplotlib.pyplot as plt
from photometry_tools import helper_func
import sep
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import matplotlib
from matplotlib.colorbar import ColorbarBase


bv_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}
bv_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}
bv_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.1], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.1], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}

bv_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.7, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [+0.05, -0.9], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
bv_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
bv_annotation_dict = {
    500: {'offset': [-0.5, +0.3], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}


def display_models(ax, y_color='bv',
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
        for age in label_dict.keys():
            ax.text(model_vi_sol[age_mod_sol == age]+label_dict[age]['offsets'][0],
                    y_model_sol[age_mod_sol == age]+label_dict[age]['offsets'][1],
                    label_dict[age]['label'], horizontalalignment=label_dict[age]['ha'], verticalalignment=label_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize)


        annotation_dict = globals()['%s_annotation_dict' % y_color]
        for age in annotation_dict.keys():

            ax.annotate(' ',
                        xy=(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data', color=age_label_color,
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkcyan', lw=3, ls='-'))
            ax.annotate(' ',
                        xy=(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data',
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkviolet', lw=3, ls='-'))
            ax.text(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                    y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1],
                    annotation_dict[age]['label'],
                    horizontalalignment=annotation_dict[age]['ha'], verticalalignment=annotation_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize, zorder=40)


def get_scale(x_points, y_points, x_lim, y_lim, map_shape, sep_table, table_index=0, n_frac=0.68, max_counts=10000):
    x_ell = sep_table['x'][table_index] / map_shape[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    y_ell = sep_table['y'][table_index] / map_shape[1] * (y_lim[0] - y_lim[1]) + y_lim[1]
    a_ell = sep_table['a'][table_index] / map_shape[0] * (x_lim[1] - x_lim[0])
    b_ell = sep_table['b'][table_index] / map_shape[1] * (y_lim[1] - y_lim[0])
    theta_ell = sep_table['theta'][table_index]

    scale = 1
    segment_content = len(x_points)
    initial_step_size = 0.5
    count = 1

    while True:

        mask_in_ell = helper_func.check_point_inside_ellipse(x_ell=x_ell, y_ell=y_ell, a_ell=a_ell*scale,
                                                             b_ell=b_ell*scale, theta_ell=theta_ell,
                                                             x_p=x_points, y_p=y_points) != 0
        ellipse_content = sum(mask_in_ell)
        frac = ellipse_content / segment_content

        if frac > n_frac:
            scale -= initial_step_size / count
        else:
            scale += initial_step_size / count

        # is_accuracy = abs(frac - n_frac)
        # print(scale, is_accuracy, frac)
        count += 1
        if count > max_counts:
            break

    return frac, scale


def plot_rescaled_sep_ellipse(ax, sep_table, map_shape, x_lim, y_lim, scale=1, table_index=0, color='k', linewidth=2):
    x = sep_table['x'][table_index] / map_shape[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    y = sep_table['y'][table_index] / map_shape[1] * (y_lim[0] - y_lim[1]) + y_lim[1]
    a = sep_table['a'][table_index] / map_shape[0] * (x_lim[1] - x_lim[0])
    b = sep_table['b'][table_index] / map_shape[1] * (y_lim[1] - y_lim[0])
    theta = sep_table['theta'][table_index]
    e = Ellipse(xy=(x, y), width=scale*a, height=scale*b, angle=theta*180/np.pi)
    e.set_edgecolor(color)
    e.set_facecolor('none')
    e.set_linewidth(linewidth)
    ax.add_artist(e)


age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')

color_vi_hum = np.load('../color_color/data_output/color_vi_hum.npy')
color_bv_hum = np.load('../color_color/data_output/color_bv_hum.npy')
color_vi_err_hum = np.load('../color_color/data_output/color_vi_err_hum.npy')
color_bv_err_hum = np.load('../color_color/data_output/color_bv_err_hum.npy')
detect_nuv_hum = np.load('../color_color/data_output/detect_nuv_hum.npy')
detect_u_hum = np.load('../color_color/data_output/detect_u_hum.npy')
detect_b_hum = np.load('../color_color/data_output/detect_b_hum.npy')
detect_v_hum = np.load('../color_color/data_output/detect_v_hum.npy')
detect_i_hum = np.load('../color_color/data_output/detect_i_hum.npy')
clcl_color_hum = np.load('../color_color/data_output/clcl_color_hum.npy')
age_hum = np.load('../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../color_color/data_output/ebv_hum.npy')

color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_bv_ml = np.load('../color_color/data_output/color_bv_ml.npy')
color_vi_err_ml = np.load('../color_color/data_output/color_vi_err_ml.npy')
color_bv_err_ml = np.load('../color_color/data_output/color_bv_err_ml.npy')
detect_nuv_ml = np.load('../color_color/data_output/detect_nuv_ml.npy')
detect_u_ml = np.load('../color_color/data_output/detect_u_ml.npy')
detect_b_ml = np.load('../color_color/data_output/detect_b_ml.npy')
detect_v_ml = np.load('../color_color/data_output/detect_v_ml.npy')
detect_i_ml = np.load('../color_color/data_output/detect_i_ml.npy')
clcl_color_ml = np.load('../color_color/data_output/clcl_color_ml.npy')
clcl_qual_color_ml = np.load('../color_color/data_output/clcl_qual_color_ml.npy')
age_ml = np.load('../color_color/data_output/age_ml.npy')
ebv_ml = np.load('../color_color/data_output/ebv_ml.npy')

# color range limitations
x_lim_vi = (-0.6, 1.9)
y_lim_bv = (1.4, -0.7)
# x_lim_vi = (-0.6, 2.5)
#
# y_lim_bv = (1.7, -0.7)

mask_detect_bvvi_hum = detect_nuv_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_bvvi_ml = detect_nuv_ml * detect_b_ml * detect_v_ml * detect_i_ml

mask_good_colors_bvvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_bv_hum > (y_lim_bv[1] - 1)) & (color_bv_hum < (y_lim_bv[0] + 1)) &
                               mask_detect_bvvi_hum)
mask_good_colors_bvvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_bv_ml > (y_lim_bv[1] - 1)) & (color_bv_ml < (y_lim_bv[0] + 1)) &
                               mask_detect_bvvi_ml)

mask_class_12_hum = (clcl_color_hum == 1) #| (clcl_color_hum == 2)
mask_class_12_ml = (clcl_color_ml == 1) #| (clcl_color_ml == 2)



# get gauss und segmentations
n_bins_bvvi = 150
threshold_fact = 4
kernal_std = 1.0
contrast = 0.1

gauss_dict_bvvi_hum_12 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_12_hum * mask_good_colors_bvvi_hum],
                                              y_data=color_bv_hum[mask_class_12_hum * mask_good_colors_bvvi_hum],
                                              x_data_err=color_vi_err_hum[mask_class_12_hum * mask_good_colors_bvvi_hum],
                                              y_data_err=color_bv_err_hum[mask_class_12_hum * mask_good_colors_bvvi_hum],
                                              x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi,
                                              threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)
gauss_dict_bvvi_ml_12 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_12_ml * mask_good_colors_bvvi_ml],
                                             y_data=color_bv_ml[mask_class_12_ml * mask_good_colors_bvvi_ml],
                                             x_data_err=color_vi_err_ml[mask_class_12_ml * mask_good_colors_bvvi_ml],
                                             y_data_err=color_bv_err_ml[mask_class_12_ml * mask_good_colors_bvvi_ml],
                                             x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi,
                                             threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)

np.save('data_output/gauss_dict_bvvi_hum_12.npy', gauss_dict_bvvi_hum_12)
np.save('data_output/gauss_dict_bvvi_ml_12.npy', gauss_dict_bvvi_ml_12)

gauss_map_hum_12 = gauss_dict_bvvi_hum_12['gauss_map']
seg_map_hum_12 = gauss_dict_bvvi_hum_12['seg_deb_map']
young_map_hum_12 = gauss_map_hum_12.copy()
cascade_map_hum_12 = gauss_map_hum_12.copy()
gc_map_hum_12 = gauss_map_hum_12.copy()
young_map_hum_12[seg_map_hum_12._data != 1] = 0
cascade_map_hum_12[seg_map_hum_12._data != 2] = 0
gc_map_hum_12[seg_map_hum_12._data != 3] = 0

gauss_map_ml_12 = gauss_dict_bvvi_ml_12['gauss_map']
seg_map_ml_12 = gauss_dict_bvvi_ml_12['seg_deb_map']
young_map_ml_12 = gauss_map_ml_12.copy()
cascade_map_ml_12 = gauss_map_ml_12.copy()
gc_map_ml_12 = gauss_map_ml_12.copy()
young_map_ml_12[seg_map_ml_12._data != 1] = 0
cascade_map_ml_12[seg_map_ml_12._data != 2] = 0
gc_map_ml_12[seg_map_ml_12._data != 3] = 0


# figure = plt.figure(figsize=(22, 14))
# fontsize = 21
#
# ax_cc_reg_hum = figure.add_axes([0.725, 0.515, 0.27, 0.46])
# ax_cc_reg_ml = figure.add_axes([0.725, 0.05, 0.27, 0.46])
#
# helper_func.plot_reg_map(ax=ax_cc_reg_hum, gauss_map=gauss_dict_bvvi_hum_12['gauss_map'], seg_map=gauss_dict_bvvi_hum_12['seg_deb_map'], smooth_kernel=kernal_std,
#                          x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi)
#
# helper_func.plot_reg_map(ax=ax_cc_reg_ml, gauss_map=gauss_dict_bvvi_ml_12['gauss_map'], seg_map=gauss_dict_bvvi_ml_12['seg_deb_map'], smooth_kernel=kernal_std,
#                          x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi)
#
# plt.show()
#
# exit()




vi_hull_young_hum_12, bv_hull_young_hum_12 = helper_func.seg2hull(seg_map_hum_12, x_lim=x_lim_vi, y_lim=y_lim_bv,
                                                                  n_bins=n_bins_bvvi, seg_index=1, contour_index=0,
                                                                  save_str='bvvi_young_hum_12', x_label='vi', y_label='bv')
vi_hull_cascade_hum_12, bv_hull_cascade_hum_12 = helper_func.seg2hull(seg_map_hum_12, x_lim=x_lim_vi, y_lim=y_lim_bv,
                                                                  n_bins=n_bins_bvvi, seg_index=2, contour_index=0,
                                                                  save_str='bvvi_cascade_hum_12', x_label='vi', y_label='bv')
# vi_hull_gc_hum_12, bv_hull_gc_hum_12 = helper_func.seg2hull(seg_map_hum_12, x_lim=x_lim_vi, y_lim=y_lim_bv,
#                                                                   n_bins=n_bins_bvvi, seg_index=3, contour_index=0,
#                                                                   save_str='bvvi_gc_hum_12', x_label='vi', y_label='bv')

vi_hull_young_ml_12, bv_hull_young_ml_12 = helper_func.seg2hull(seg_map_ml_12, x_lim=x_lim_vi, y_lim=y_lim_bv,
                                                                  n_bins=n_bins_bvvi, seg_index=1, contour_index=0,
                                                                  save_str='bvvi_young_ml_12', x_label='vi', y_label='bv')
vi_hull_cascade_ml_12, bv_hull_cascade_ml_12 = helper_func.seg2hull(seg_map_ml_12, x_lim=x_lim_vi, y_lim=y_lim_bv,
                                                                  n_bins=n_bins_bvvi, seg_index=2, contour_index=0,
                                                                  save_str='bvvi_cascade_ml_12', x_label='vi', y_label='bv')
# vi_hull_gc_ml_12, bv_hull_gc_ml_12 = helper_func.seg2hull(seg_map_ml_12, x_lim=x_lim_vi, y_lim=y_lim_bv,
#                                                                   n_bins=n_bins_bvvi, seg_index=3, contour_index=0,
#                                                                   save_str='bvvi_gc_ml_12', x_label='vi', y_label='bv')

hull_bvvi_young_hum_12 = ConvexHull(np.array([vi_hull_young_hum_12, bv_hull_young_hum_12]).T)
hull_bvvi_cascade_hum_12 = ConvexHull(np.array([vi_hull_cascade_hum_12, bv_hull_cascade_hum_12]).T)
# hull_bvvi_gc_hum_12 = ConvexHull(np.array([vi_hull_gc_hum_12, bv_hull_gc_hum_12]).T)
hull_bvvi_young_ml_12 = ConvexHull(np.array([vi_hull_young_ml_12, bv_hull_young_ml_12]).T)
hull_bvvi_cascade_ml_12 = ConvexHull(np.array([vi_hull_cascade_ml_12, bv_hull_cascade_ml_12]).T)
# hull_bvvi_gc_ml_12 = ConvexHull(np.array([vi_hull_gc_ml_12, bv_hull_gc_ml_12]).T)

in_hull_young_hum_12 = helper_func.points_in_hull(np.array([color_vi_hum, color_bv_hum]).T, hull_bvvi_young_hum_12)
in_hull_cascade_hum_12 = helper_func.points_in_hull(np.array([color_vi_hum, color_bv_hum]).T, hull_bvvi_cascade_hum_12)
# in_hull_gc_hum_12 = helper_func.points_in_hull(np.array([color_vi_hum, color_bv_hum]).T, hull_bvvi_gc_hum_12)
in_hull_young_ml_12 = helper_func.points_in_hull(np.array([color_vi_ml, color_bv_ml]).T, hull_bvvi_young_ml_12)
in_hull_cascade_ml_12 = helper_func.points_in_hull(np.array([color_vi_ml, color_bv_ml]).T, hull_bvvi_cascade_ml_12)
# in_hull_gc_ml_12 = helper_func.points_in_hull(np.array([color_vi_ml, color_bv_ml]).T, hull_bvvi_gc_ml_12)


# bins
n_bins_uncertainty_bvvi = 50
x_bins = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_uncertainty_bvvi)
y_bins = np.linspace(y_lim_bv[1], y_lim_bv[0], n_bins_uncertainty_bvvi)
threshold_hum = 5
threshold_ml = 5


snr_vi_map_hum = np.zeros((len(x_bins), len(y_bins))) * np.nan
snr_vi_map_ml = np.zeros((len(x_bins), len(y_bins))) * np.nan
snr_bv_map_hum = np.zeros((len(x_bins), len(y_bins))) * np.nan
snr_bv_map_ml = np.zeros((len(x_bins), len(y_bins))) * np.nan


for x_index in range(len(x_bins)-1):
    for y_index in range(len(y_bins)-1):
        mask_in_bin_hum = ((color_vi_hum > x_bins[x_index]) & (color_vi_hum < x_bins[x_index + 1]) &
                           (color_bv_hum > y_bins[y_index]) & (color_bv_hum < y_bins[y_index + 1]))
        mask_in_bin_ml = ((color_vi_ml > x_bins[x_index]) & (color_vi_ml < x_bins[x_index + 1]) &
                          (color_bv_ml > y_bins[y_index]) & (color_bv_ml < y_bins[y_index + 1]))

        mask_selected_obj_hum = mask_in_bin_hum * mask_class_12_hum * mask_good_colors_bvvi_hum
        mask_selected_obj_ml = mask_in_bin_ml * mask_class_12_ml * mask_good_colors_bvvi_ml

        if sum(mask_selected_obj_hum) > threshold_hum:
            snr_vi_map_hum[x_index, y_index] = np.nanmean(color_vi_err_hum[mask_selected_obj_hum])
            snr_bv_map_hum[x_index, y_index] = np.nanmean(color_bv_err_hum[mask_selected_obj_hum])
        if sum(mask_selected_obj_ml) > threshold_ml:
            snr_vi_map_ml[x_index, y_index] = np.nanmean(color_vi_err_ml[mask_selected_obj_ml])
            snr_bv_map_ml[x_index, y_index] = np.nanmean(color_bv_err_ml[mask_selected_obj_ml])


figure = plt.figure(figsize=(22, 14))
fontsize = 21

cmap_vi = matplotlib.cm.get_cmap('inferno')
norm_vi = matplotlib.colors.Normalize(vmin=0, vmax=0.11)

cmap_bv = matplotlib.cm.get_cmap('cividis')
norm_bv = matplotlib.colors.Normalize(vmin=0, vmax=0.11)

ax_cc_sn_vi_hum = figure.add_axes([0.055, 0.515, 0.27, 0.46])
ax_cc_sn_vi_ml = figure.add_axes([0.055, 0.05, 0.27, 0.46])

ax_cc_sn_bv_hum = figure.add_axes([0.395, 0.515, 0.27, 0.46])
ax_cc_sn_bv_ml = figure.add_axes([0.395, 0.05, 0.27, 0.46])

ax_cc_reg_hum = figure.add_axes([0.725, 0.515, 0.27, 0.46])
ax_cc_reg_ml = figure.add_axes([0.725, 0.05, 0.27, 0.46])

ax_cbar_sn_vi = figure.add_axes([0.33, 0.2, 0.015, 0.6])
ax_cbar_sn_bv = figure.add_axes([0.67, 0.2, 0.015, 0.6])


ax_cc_sn_vi_hum.imshow(snr_vi_map_hum.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
                       cmap=cmap_vi, norm=norm_vi,  interpolation='nearest', aspect='auto')
ax_cc_sn_vi_ml.imshow(snr_vi_map_ml.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
                      cmap=cmap_vi, norm=norm_vi,  interpolation='nearest', aspect='auto')
ax_cc_sn_bv_hum.imshow(snr_bv_map_hum.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
                       cmap=cmap_bv, norm=norm_bv,  interpolation='nearest', aspect='auto')
ax_cc_sn_bv_ml.imshow(snr_bv_map_ml.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
                      cmap=cmap_bv, norm=norm_bv,  interpolation='nearest', aspect='auto')


display_models(ax=ax_cc_sn_vi_hum, y_color='bv', label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax_cc_sn_vi_ml, y_color='bv')
display_models(ax=ax_cc_sn_bv_hum, y_color='bv')
display_models(ax=ax_cc_sn_bv_ml, y_color='bv')
display_models(ax=ax_cc_reg_hum, y_color='bv')
display_models(ax=ax_cc_reg_ml, y_color='bv')


ColorbarBase(ax_cbar_sn_vi, orientation='vertical', cmap=cmap_vi, norm=norm_vi, extend='neither', ticks=None)
ax_cbar_sn_vi.set_ylabel(r'$\sigma_{\rm V-I}$', labelpad=-4, fontsize=fontsize + 5)
ax_cbar_sn_vi.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)

ColorbarBase(ax_cbar_sn_bv, orientation='vertical', cmap=cmap_bv, norm=norm_bv, extend='neither', ticks=None)
ax_cbar_sn_bv.set_ylabel(r'$\sigma_{\rm B-V}$', labelpad=-4, fontsize=fontsize + 5)
ax_cbar_sn_bv.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)


helper_func.plot_reg_map(ax=ax_cc_reg_hum, gauss_map=gauss_dict_bvvi_hum_12['gauss_map'], seg_map=gauss_dict_bvvi_hum_12['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi)

helper_func.plot_reg_map(ax=ax_cc_reg_ml, gauss_map=gauss_dict_bvvi_ml_12['gauss_map'], seg_map=gauss_dict_bvvi_ml_12['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi)

vi_int = 1.3
bv_int = -0.2
av_value = 1



helper_func.plot_reddening_vect(ax=ax_cc_sn_vi_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                                x_text_offset=0.00, y_text_offset=-0.1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-3)
helper_func.plot_reddening_vect(ax=ax_cc_sn_vi_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

helper_func.plot_reddening_vect(ax=ax_cc_sn_bv_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_sn_bv_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


helper_func.plot_reddening_vect(ax=ax_cc_reg_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_reg_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


ax_cc_sn_vi_hum.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_sn_vi_ml.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_sn_bv_hum.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_sn_bv_ml.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_hum.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_ml.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

ax_cc_sn_vi_hum.set_title('Mean V-I uncertainty map', fontsize=fontsize)
ax_cc_sn_bv_hum.set_title('Mean NUV-B uncertainty map', fontsize=fontsize)
ax_cc_reg_hum.set_title('Uncertainty weighted density', fontsize=fontsize)

ax_cc_sn_vi_hum.set_xlim(x_lim_vi)
ax_cc_sn_vi_ml.set_xlim(x_lim_vi)
ax_cc_sn_bv_hum.set_xlim(x_lim_vi)
ax_cc_sn_bv_ml.set_xlim(x_lim_vi)
ax_cc_reg_hum.set_xlim(x_lim_vi)
ax_cc_reg_ml.set_xlim(x_lim_vi)

ax_cc_sn_vi_hum.set_ylim(y_lim_bv)
ax_cc_sn_vi_ml.set_ylim(y_lim_bv)
ax_cc_sn_bv_hum.set_ylim(y_lim_bv)
ax_cc_sn_bv_ml.set_ylim(y_lim_bv)
ax_cc_reg_hum.set_ylim(y_lim_bv)
ax_cc_reg_ml.set_ylim(y_lim_bv)

ax_cc_reg_hum.set_yticklabels([])
ax_cc_reg_ml.set_yticklabels([])
ax_cc_sn_bv_hum.set_yticklabels([])
ax_cc_sn_bv_ml.set_yticklabels([])

ax_cc_sn_vi_hum.set_xticklabels([])
ax_cc_sn_bv_hum.set_xticklabels([])
ax_cc_reg_hum.set_xticklabels([])


ax_cc_sn_vi_hum.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_cc_sn_vi_ml.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_cc_sn_vi_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_sn_bv_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_reg_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_cc_sn_vi_hum.legend(frameon=False, loc=3, fontsize=fontsize)

ax_cc_sn_vi_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_vi_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_bv_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_bv_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_reg_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_reg_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.savefig('plot_output/uncert_reg_c12_bvvi.png')
plt.savefig('plot_output/uncert_reg_c12_bvvi.pdf')




