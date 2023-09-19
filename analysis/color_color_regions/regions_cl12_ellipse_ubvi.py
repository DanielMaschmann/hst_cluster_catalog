import numpy as np
import matplotlib.pyplot as plt
from photometry_tools import helper_func
import sep
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import matplotlib
from matplotlib.colorbar import ColorbarBase


nuvb_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}
ub_label_dict = {
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
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')

color_vi_hum = np.load('../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../color_color/data_output/color_ub_hum.npy')
color_vi_err_hum = np.load('../color_color/data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('../color_color/data_output/color_ub_err_hum.npy')
detect_u_hum = np.load('../color_color/data_output/detect_u_hum.npy')
detect_b_hum = np.load('../color_color/data_output/detect_b_hum.npy')
detect_v_hum = np.load('../color_color/data_output/detect_v_hum.npy')
detect_i_hum = np.load('../color_color/data_output/detect_i_hum.npy')
clcl_color_hum = np.load('../color_color/data_output/clcl_color_hum.npy')
age_hum = np.load('../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../color_color/data_output/ebv_hum.npy')

color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
color_vi_err_ml = np.load('../color_color/data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('../color_color/data_output/color_ub_err_ml.npy')
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
y_lim_ub = (0.9, -1.9)

mask_class_12_hum = (clcl_color_hum == 1) | (clcl_color_hum == 2)
mask_class_12_ml = (clcl_color_ml == 1) | (clcl_color_ml == 2)

mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                             (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                             mask_detect_ubvi_hum)
mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                            (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                            mask_detect_ubvi_ml)


# get gauss und segmentations
n_bins_ubvi = 120
threshold_fact = 3
kernal_std = 1.0
contrast = 0.01

gauss_dict_ubvi_hum_12 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              y_data=color_ub_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              x_data_err=color_vi_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              y_data_err=color_ub_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_ml_12 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_12_ml * mask_good_colors_ubvi_ml],
                                             y_data=color_ub_ml[mask_class_12_ml * mask_good_colors_ubvi_ml],
                                             x_data_err=color_vi_err_ml[mask_class_12_ml * mask_good_colors_ubvi_ml],
                                             y_data_err=color_ub_err_ml[mask_class_12_ml * mask_good_colors_ubvi_ml],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

np.save('data_output/gauss_dict_ubvi_hum_12.npy', gauss_dict_ubvi_hum_12)
np.save('data_output/gauss_dict_ubvi_ml_12.npy', gauss_dict_ubvi_ml_12)

gauss_map_hum_12 = gauss_dict_ubvi_hum_12['gauss_map']
seg_map_hum_12 = gauss_dict_ubvi_hum_12['seg_deb_map']
young_map_hum_12 = gauss_map_hum_12.copy()
cascade_map_hum_12 = gauss_map_hum_12.copy()
gc_map_hum_12 = gauss_map_hum_12.copy()
young_map_hum_12[seg_map_hum_12._data != 1] = 0
cascade_map_hum_12[seg_map_hum_12._data != 2] = 0
gc_map_hum_12[seg_map_hum_12._data != 3] = 0

gauss_map_ml_12 = gauss_dict_ubvi_ml_12['gauss_map']
seg_map_ml_12 = gauss_dict_ubvi_ml_12['seg_deb_map']
young_map_ml_12 = gauss_map_ml_12.copy()
cascade_map_ml_12 = gauss_map_ml_12.copy()
gc_map_ml_12 = gauss_map_ml_12.copy()
young_map_ml_12[seg_map_ml_12._data != 1] = 0
cascade_map_ml_12[seg_map_ml_12._data != 2] = 0
gc_map_ml_12[seg_map_ml_12._data != 3] = 0

vi_hull_young_hum_12, ub_hull_young_hum_12 = helper_func.seg2hull(seg_map_hum_12, x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                  n_bins=n_bins_ubvi, seg_index=1, contour_index=0,
                                                                  save_str='ubvi_young_hum_12', x_label='vi', y_label='ub')
vi_hull_cascade_hum_12, ub_hull_cascade_hum_12 = helper_func.seg2hull(seg_map_hum_12, x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                  n_bins=n_bins_ubvi, seg_index=2, contour_index=0,
                                                                  save_str='ubvi_cascade_hum_12', x_label='vi', y_label='ub')
vi_hull_gc_hum_12, ub_hull_gc_hum_12 = helper_func.seg2hull(seg_map_hum_12, x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                  n_bins=n_bins_ubvi, seg_index=3, contour_index=0,
                                                                  save_str='ubvi_gc_hum_12', x_label='vi', y_label='ub')

vi_hull_young_ml_12, ub_hull_young_ml_12 = helper_func.seg2hull(seg_map_ml_12, x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                  n_bins=n_bins_ubvi, seg_index=1, contour_index=0,
                                                                  save_str='ubvi_young_ml_12', x_label='vi', y_label='ub')
vi_hull_cascade_ml_12, ub_hull_cascade_ml_12 = helper_func.seg2hull(seg_map_ml_12, x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                  n_bins=n_bins_ubvi, seg_index=2, contour_index=0,
                                                                  save_str='ubvi_cascade_ml_12', x_label='vi', y_label='ub')
vi_hull_gc_ml_12, ub_hull_gc_ml_12 = helper_func.seg2hull(seg_map_ml_12, x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                  n_bins=n_bins_ubvi, seg_index=3, contour_index=0,
                                                                  save_str='ubvi_gc_ml_12', x_label='vi', y_label='ub')

hull_ubvi_young_hum_12 = ConvexHull(np.array([vi_hull_young_hum_12, ub_hull_young_hum_12]).T)
hull_ubvi_cascade_hum_12 = ConvexHull(np.array([vi_hull_cascade_hum_12, ub_hull_cascade_hum_12]).T)
hull_ubvi_gc_hum_12 = ConvexHull(np.array([vi_hull_gc_hum_12, ub_hull_gc_hum_12]).T)
hull_ubvi_young_ml_12 = ConvexHull(np.array([vi_hull_young_ml_12, ub_hull_young_ml_12]).T)
hull_ubvi_cascade_ml_12 = ConvexHull(np.array([vi_hull_cascade_ml_12, ub_hull_cascade_ml_12]).T)
hull_ubvi_gc_ml_12 = ConvexHull(np.array([vi_hull_gc_ml_12, ub_hull_gc_ml_12]).T)

in_hull_young_hum_12 = helper_func.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_ubvi_young_hum_12)
in_hull_cascade_hum_12 = helper_func.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_ubvi_cascade_hum_12)
in_hull_gc_hum_12 = helper_func.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_ubvi_gc_hum_12)
in_hull_young_ml_12 = helper_func.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_ubvi_young_ml_12)
in_hull_cascade_ml_12 = helper_func.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_ubvi_cascade_ml_12)
in_hull_gc_ml_12 = helper_func.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_ubvi_gc_ml_12)

sep_table_young_hum_12 = sep.extract(data=young_map_hum_12, thresh=np.nanmax(young_map_hum_12)/100)
sep_table_cascade_hum_12 = sep.extract(data=cascade_map_hum_12, thresh=np.nanmax(cascade_map_hum_12)/100)
sep_table_gc_hum_12 = sep.extract(data=gc_map_hum_12, thresh=np.nanmax(gc_map_hum_12)/100)

sep_table_young_ml_12 = sep.extract(data=young_map_ml_12, thresh=np.nanmax(young_map_ml_12)/100)
sep_table_cascade_ml_12 = sep.extract(data=cascade_map_ml_12, thresh=np.nanmax(cascade_map_ml_12)/100)
sep_table_gc_ml_12 = sep.extract(data=gc_map_ml_12, thresh=np.nanmax(gc_map_ml_12)/100)

frac_young_hum_12, scale_young_hum_12 = get_scale(
    x_points=color_vi_hum[in_hull_young_hum_12 * mask_class_12_hum * mask_good_colors_ubvi_hum],
    y_points=color_ub_hum[in_hull_young_hum_12 * mask_class_12_hum * mask_good_colors_ubvi_hum],
    x_lim=x_lim_vi, y_lim=y_lim_ub, map_shape=young_map_hum_12.shape, sep_table=sep_table_young_hum_12,
    table_index=0, n_frac=0.68, max_counts=10000)
frac_cascade_hum_12, scale_cascade_hum_12 = get_scale(
    x_points=color_vi_hum[in_hull_cascade_hum_12 * mask_class_12_hum * mask_good_colors_ubvi_hum],
    y_points=color_ub_hum[in_hull_cascade_hum_12 * mask_class_12_hum * mask_good_colors_ubvi_hum],
    x_lim=x_lim_vi, y_lim=y_lim_ub, map_shape=cascade_map_hum_12.shape, sep_table=sep_table_cascade_hum_12,
    table_index=0, n_frac=0.68, max_counts=10000)
frac_gc_hum_12, scale_gc_hum_12 = get_scale(
    x_points=color_vi_hum[in_hull_gc_hum_12 * mask_class_12_hum * mask_good_colors_ubvi_hum],
    y_points=color_ub_hum[in_hull_gc_hum_12 * mask_class_12_hum * mask_good_colors_ubvi_hum],
    x_lim=x_lim_vi, y_lim=y_lim_ub, map_shape=gc_map_hum_12.shape, sep_table=sep_table_gc_hum_12,
    table_index=0, n_frac=0.68, max_counts=10000)

frac_young_ml_12, scale_young_ml_12 = get_scale(
    x_points=color_vi_ml[in_hull_young_ml_12 * mask_class_12_ml * mask_good_colors_ubvi_ml],
    y_points=color_ub_ml[in_hull_young_ml_12 * mask_class_12_ml * mask_good_colors_ubvi_ml],
    x_lim=x_lim_vi, y_lim=y_lim_ub, map_shape=young_map_ml_12.shape, sep_table=sep_table_young_ml_12,
    table_index=0, n_frac=0.68, max_counts=10000)
frac_cascade_ml_12, scale_cascade_ml_12 = get_scale(
    x_points=color_vi_ml[in_hull_cascade_ml_12 * mask_class_12_ml * mask_good_colors_ubvi_ml],
    y_points=color_ub_ml[in_hull_cascade_ml_12 * mask_class_12_ml * mask_good_colors_ubvi_ml],
    x_lim=x_lim_vi, y_lim=y_lim_ub, map_shape=cascade_map_ml_12.shape, sep_table=sep_table_cascade_ml_12,
    table_index=0, n_frac=0.68, max_counts=10000)
frac_gc_ml_12, scale_gc_ml_12 = get_scale(
    x_points=color_vi_ml[in_hull_gc_ml_12 * mask_class_12_ml * mask_good_colors_ubvi_ml],
    y_points=color_ub_ml[in_hull_gc_ml_12 * mask_class_12_ml * mask_good_colors_ubvi_ml],
    x_lim=x_lim_vi, y_lim=y_lim_ub, map_shape=gc_map_ml_12.shape, sep_table=sep_table_gc_ml_12,
    table_index=0, n_frac=0.68, max_counts=10000)


# bins
n_bins_uncertainty_ubvi = 50
x_bins = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_uncertainty_ubvi)
y_bins = np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins_uncertainty_ubvi)
threshold_hum = 5
threshold_ml = 5


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

        mask_selected_obj_hum = mask_in_bin_hum * mask_class_12_hum * mask_good_colors_ubvi_hum
        mask_selected_obj_ml = mask_in_bin_ml * mask_class_12_ml * mask_good_colors_ubvi_ml

        if sum(mask_selected_obj_hum) > threshold_hum:
            snr_vi_map_hum[x_index, y_index] = np.nanmean(color_vi_err_hum[mask_selected_obj_hum])
            snr_ub_map_hum[x_index, y_index] = np.nanmean(color_ub_err_hum[mask_selected_obj_hum])
        if sum(mask_selected_obj_ml) > threshold_ml:
            snr_vi_map_ml[x_index, y_index] = np.nanmean(color_vi_err_ml[mask_selected_obj_ml])
            snr_ub_map_ml[x_index, y_index] = np.nanmean(color_ub_err_ml[mask_selected_obj_ml])


figure = plt.figure(figsize=(22, 14))
fontsize = 21

cmap_vi = matplotlib.cm.get_cmap('inferno')
norm_vi = matplotlib.colors.Normalize(vmin=0, vmax=0.11)

cmap_ub = matplotlib.cm.get_cmap('cividis')
norm_ub = matplotlib.colors.Normalize(vmin=0, vmax=0.5)

ax_cc_sn_vi_hum = figure.add_axes([0.055, 0.515, 0.27, 0.46])
ax_cc_sn_vi_ml = figure.add_axes([0.055, 0.05, 0.27, 0.46])

ax_cc_sn_ub_hum = figure.add_axes([0.395, 0.515, 0.27, 0.46])
ax_cc_sn_ub_ml = figure.add_axes([0.395, 0.05, 0.27, 0.46])

ax_cc_reg_hum = figure.add_axes([0.725, 0.515, 0.27, 0.46])
ax_cc_reg_ml = figure.add_axes([0.725, 0.05, 0.27, 0.46])

ax_cbar_sn_vi = figure.add_axes([0.33, 0.2, 0.015, 0.6])
ax_cbar_sn_ub = figure.add_axes([0.67, 0.2, 0.015, 0.6])


ax_cc_sn_vi_hum.imshow(snr_vi_map_hum.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                       cmap=cmap_vi, norm=norm_vi,  interpolation='nearest', aspect='auto')
ax_cc_sn_vi_ml.imshow(snr_vi_map_ml.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                      cmap=cmap_vi, norm=norm_vi,  interpolation='nearest', aspect='auto')
ax_cc_sn_ub_hum.imshow(snr_ub_map_hum.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                       cmap=cmap_ub, norm=norm_ub,  interpolation='nearest', aspect='auto')
ax_cc_sn_ub_ml.imshow(snr_ub_map_ml.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                      cmap=cmap_ub, norm=norm_ub,  interpolation='nearest', aspect='auto')


display_models(ax=ax_cc_sn_vi_hum, y_color='ub', label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax_cc_sn_vi_ml, y_color='ub')
display_models(ax=ax_cc_sn_ub_hum, y_color='ub')
display_models(ax=ax_cc_sn_ub_ml, y_color='ub')
display_models(ax=ax_cc_reg_hum, y_color='ub')
display_models(ax=ax_cc_reg_ml, y_color='ub')


ColorbarBase(ax_cbar_sn_vi, orientation='vertical', cmap=cmap_vi, norm=norm_vi, extend='neither', ticks=None)
ax_cbar_sn_vi.set_ylabel(r'$\sigma_{\rm V-I}$', labelpad=-4, fontsize=fontsize + 5)
ax_cbar_sn_vi.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)

ColorbarBase(ax_cbar_sn_ub, orientation='vertical', cmap=cmap_ub, norm=norm_ub, extend='neither', ticks=None)
ax_cbar_sn_ub.set_ylabel(r'$\sigma_{\rm U-B}$', labelpad=-4, fontsize=fontsize + 5)
ax_cbar_sn_ub.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)


helper_func.plot_reg_map(ax=ax_cc_reg_hum, gauss_map=gauss_dict_ubvi_hum_12['gauss_map'], seg_map=gauss_dict_ubvi_hum_12['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)

helper_func.plot_reg_map(ax=ax_cc_reg_ml, gauss_map=gauss_dict_ubvi_ml_12['gauss_map'], seg_map=gauss_dict_ubvi_ml_12['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)

plot_rescaled_sep_ellipse(ax=ax_cc_reg_hum, sep_table=sep_table_young_hum_12, map_shape=young_map_hum_12.shape,
                          x_lim=x_lim_vi, y_lim=y_lim_ub, scale=scale_young_hum_12, table_index=0, color='k')
plot_rescaled_sep_ellipse(ax=ax_cc_reg_hum, sep_table=sep_table_cascade_hum_12, map_shape=cascade_map_hum_12.shape,
                          x_lim=x_lim_vi, y_lim=y_lim_ub, scale=scale_cascade_hum_12, table_index=0, color='k')
plot_rescaled_sep_ellipse(ax=ax_cc_reg_hum, sep_table=sep_table_gc_hum_12, map_shape=gc_map_hum_12.shape,
                          x_lim=x_lim_vi, y_lim=y_lim_ub, scale=scale_gc_hum_12, table_index=0, color='k')

plot_rescaled_sep_ellipse(ax=ax_cc_reg_ml, sep_table=sep_table_young_ml_12, map_shape=young_map_ml_12.shape,
                          x_lim=x_lim_vi, y_lim=y_lim_ub, scale=scale_young_ml_12, table_index=0, color='k')
plot_rescaled_sep_ellipse(ax=ax_cc_reg_ml, sep_table=sep_table_cascade_ml_12, map_shape=cascade_map_ml_12.shape,
                          x_lim=x_lim_vi, y_lim=y_lim_ub, scale=scale_cascade_ml_12, table_index=0, color='k')
plot_rescaled_sep_ellipse(ax=ax_cc_reg_ml, sep_table=sep_table_gc_ml_12, map_shape=gc_map_ml_12.shape,
                          x_lim=x_lim_vi, y_lim=y_lim_ub, scale=scale_gc_ml_12, table_index=0, color='k')

ax_cc_reg_hum.scatter(sep_table_young_hum_12['x'][0] / young_map_hum_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0],
                      sep_table_young_hum_12['y'][0] / young_map_hum_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1],
                      color='k', edgecolor='k', s=80)
ax_cc_reg_hum.scatter(sep_table_cascade_hum_12['x'][0] / cascade_map_hum_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0],
                      sep_table_cascade_hum_12['y'][0] / cascade_map_hum_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1],
                      color='k', edgecolor='k', s=80)
ax_cc_reg_hum.scatter(sep_table_gc_hum_12['x'][0] / gc_map_hum_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0],
                      sep_table_gc_hum_12['y'][0] / gc_map_hum_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1],
                      color='k', edgecolor='k', s=80)

ax_cc_reg_ml.scatter(sep_table_young_ml_12['x'][0] / young_map_ml_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0],
                      sep_table_young_ml_12['y'][0] / young_map_ml_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1],
                      color='k', edgecolor='k', s=80)
ax_cc_reg_ml.scatter(sep_table_cascade_ml_12['x'][0] / cascade_map_ml_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0],
                      sep_table_cascade_ml_12['y'][0] / cascade_map_ml_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1],
                      color='k', edgecolor='k', s=80)
ax_cc_reg_ml.scatter(sep_table_gc_ml_12['x'][0] / gc_map_ml_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0],
                      sep_table_gc_ml_12['y'][0] / gc_map_ml_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1],
                      color='k', edgecolor='k', s=80)


vi_int = 1.2
ub_int = -1.4
av_value = 1

helper_func.plot_reddening_vect(ax=ax_cc_sn_vi_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.00, y_text_offset=-0.1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-3)
helper_func.plot_reddening_vect(ax=ax_cc_sn_vi_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

helper_func.plot_reddening_vect(ax=ax_cc_sn_ub_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_sn_ub_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


helper_func.plot_reddening_vect(ax=ax_cc_reg_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_reg_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


ax_cc_sn_vi_hum.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_sn_vi_ml.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_sn_ub_hum.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_sn_ub_ml.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_hum.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_ml.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

ax_cc_sn_vi_hum.set_title('Mean V-I uncertainty map', fontsize=fontsize)
ax_cc_sn_ub_hum.set_title('Mean U-B uncertainty map', fontsize=fontsize)
ax_cc_reg_hum.set_title('Uncertainty weighted density', fontsize=fontsize)

ax_cc_sn_vi_hum.set_xlim(x_lim_vi)
ax_cc_sn_vi_ml.set_xlim(x_lim_vi)
ax_cc_sn_ub_hum.set_xlim(x_lim_vi)
ax_cc_sn_ub_ml.set_xlim(x_lim_vi)
ax_cc_reg_hum.set_xlim(x_lim_vi)
ax_cc_reg_ml.set_xlim(x_lim_vi)

ax_cc_sn_vi_hum.set_ylim(y_lim_ub)
ax_cc_sn_vi_ml.set_ylim(y_lim_ub)
ax_cc_sn_ub_hum.set_ylim(y_lim_ub)
ax_cc_sn_ub_ml.set_ylim(y_lim_ub)
ax_cc_reg_hum.set_ylim(y_lim_ub)
ax_cc_reg_ml.set_ylim(y_lim_ub)

ax_cc_reg_hum.set_yticklabels([])
ax_cc_reg_ml.set_yticklabels([])
ax_cc_sn_ub_hum.set_yticklabels([])
ax_cc_sn_ub_ml.set_yticklabels([])

ax_cc_sn_vi_hum.set_xticklabels([])
ax_cc_sn_ub_hum.set_xticklabels([])
ax_cc_reg_hum.set_xticklabels([])


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

plt.savefig('plot_output/uncert_reg_c12_ubvi.png')
plt.savefig('plot_output/uncert_reg_c12_ubvi.pdf')


# plot table
x_young_hum = sep_table_young_hum_12['x'][0] / young_map_hum_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0]
y_young_hum = sep_table_young_hum_12['y'][0] / young_map_hum_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1]
a_young_hum = (sep_table_young_hum_12['a'][0] / young_map_hum_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0])) * scale_young_hum_12
b_young_hum = (sep_table_young_hum_12['b'][0] / young_map_hum_12.shape[1] * (y_lim_ub[1] - y_lim_ub[0])) * scale_young_hum_12
theta_young_hum = (sep_table_young_hum_12['theta'][0])*180/np.pi

x_cascade_hum = sep_table_cascade_hum_12['x'][0] / cascade_map_hum_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0]
y_cascade_hum = sep_table_cascade_hum_12['y'][0] / cascade_map_hum_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1]
a_cascade_hum = (sep_table_cascade_hum_12['a'][0] / cascade_map_hum_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0])) * scale_cascade_hum_12
b_cascade_hum = (sep_table_cascade_hum_12['b'][0] / cascade_map_hum_12.shape[1] * (y_lim_ub[1] - y_lim_ub[0])) * scale_cascade_hum_12
theta_cascade_hum = (sep_table_cascade_hum_12['theta'][0])*180/np.pi

x_gc_hum = sep_table_gc_hum_12['x'][0] / gc_map_hum_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0]
y_gc_hum = sep_table_gc_hum_12['y'][0] / gc_map_hum_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1]
a_gc_hum = (sep_table_gc_hum_12['a'][0] / gc_map_hum_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0])) * scale_gc_hum_12
b_gc_hum = (sep_table_gc_hum_12['b'][0] / gc_map_hum_12.shape[1] * (y_lim_ub[1] - y_lim_ub[0])) * scale_gc_hum_12
theta_gc_hum = (sep_table_gc_hum_12['theta'][0])*180/np.pi


x_young_ml = sep_table_young_ml_12['x'][0] / young_map_ml_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0]
y_young_ml = sep_table_young_ml_12['y'][0] / young_map_ml_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1]
a_young_ml = (sep_table_young_ml_12['a'][0] / young_map_ml_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0])) * scale_young_ml_12
b_young_ml = (sep_table_young_ml_12['b'][0] / young_map_ml_12.shape[1] * (y_lim_ub[1] - y_lim_ub[0])) * scale_young_ml_12
theta_young_ml = (sep_table_young_ml_12['theta'][0])*180/np.pi

x_cascade_ml = sep_table_cascade_ml_12['x'][0] / cascade_map_ml_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0]
y_cascade_ml = sep_table_cascade_ml_12['y'][0] / cascade_map_ml_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1]
a_cascade_ml = (sep_table_cascade_ml_12['a'][0] / cascade_map_ml_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0])) * scale_cascade_ml_12
b_cascade_ml = (sep_table_cascade_ml_12['b'][0] / cascade_map_ml_12.shape[1] * (y_lim_ub[1] - y_lim_ub[0])) * scale_cascade_ml_12
theta_cascade_ml = (sep_table_cascade_ml_12['theta'][0])*180/np.pi

x_gc_ml = sep_table_gc_ml_12['x'][0] / gc_map_ml_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0]
y_gc_ml = sep_table_gc_ml_12['y'][0] / gc_map_ml_12.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1]
a_gc_ml = (sep_table_gc_ml_12['a'][0] / gc_map_ml_12.shape[0] * (x_lim_vi[1] - x_lim_vi[0])) * scale_gc_ml_12
b_gc_ml = (sep_table_gc_ml_12['b'][0] / gc_map_ml_12.shape[1] * (y_lim_ub[1] - y_lim_ub[0])) * scale_gc_ml_12
theta_gc_ml = (sep_table_gc_ml_12['theta'][0])*180/np.pi

if theta_young_hum < 0:
    theta_young_hum += 180
if theta_cascade_hum < 0:
    theta_cascade_hum += 180
if theta_gc_hum < 0:
    theta_gc_hum += 180
if theta_young_ml < 0:
    theta_young_ml += 180
if theta_gc_ml < 0:
    theta_gc_ml += 180


print("")
print('\multicolumn{1}{c}{Region} & '
      '\multicolumn{2}{c}{Barycenter} & '
      '\multicolumn{2}{c}{Ellp. axis} & '
      '\multicolumn{1}{c}{angle} \\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{ } & '
      '\multicolumn{1}{c}{V-I} & '
      '\multicolumn{1}{c}{U-B} & '
      '\multicolumn{1}{c}{V-I} & '
      '\multicolumn{1}{c}{U-B} & '
      '\multicolumn{1}{c}{$\\theta$} \\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{ } & '
      '\multicolumn{1}{c}{mag} & '
      '\multicolumn{1}{c}{mag} & '
      '\multicolumn{1}{c}{mag} & '
      '\multicolumn{1}{c}{mag} & '
      '\multicolumn{1}{c}{degree} \\\\ ')
print('\\hline')
print('YDP$_{\\rm Hum}$ & %.2f & %.2f & %.2f & %.2f & %.1f \\\\' %
      (x_young_hum, y_young_hum, a_young_hum, b_young_hum, theta_young_hum))
print('MAC$_{\\rm Hum}$ & %.2f & %.2f & %.2f & %.2f & %.1f \\\\' %
      (x_cascade_hum, y_cascade_hum, a_cascade_hum, b_cascade_hum, theta_cascade_hum))
print('OGC$_{\\rm Hum}$ & %.2f & %.2f & %.2f & %.2f & %.1f \\\\' %
      (x_gc_hum, y_gc_hum, a_gc_hum, b_gc_hum, theta_gc_hum))
print('YDP$_{\\rm ML}$ & %.2f & %.2f & %.2f & %.2f & %.1f \\\\' %
      (x_young_ml, y_young_ml, a_young_ml, b_young_ml, theta_young_ml))
print('MAC$_{\\rm ML}$ & %.2f & %.2f & %.2f & %.2f & %.1f \\\\' %
      (x_cascade_ml, y_cascade_ml, a_cascade_ml, b_cascade_ml, theta_cascade_ml))
print('OGC$_{\\rm ML}$ & %.2f & %.2f & %.2f & %.2f & %.1f \\\\' %
      (x_gc_ml, y_gc_ml, a_gc_ml, b_gc_ml, theta_gc_ml))




