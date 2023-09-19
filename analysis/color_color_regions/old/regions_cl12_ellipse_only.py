import numpy as np
import matplotlib.pyplot as plt
from photometry_tools import helper_func
import sep
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse


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

        is_accuracy = abs(frac - n_frac)
        # print(scale, is_accuracy, frac)
        count += 1
        if count > max_counts:
            break

    # ax.scatter(color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull][mask_in_ell], color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull][mask_in_ell], c='r')
    # ax.scatter(color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull][~mask_in_ell], color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull][~mask_in_ell], c='b')

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



figure = plt.figure(figsize=(18, 10.5))
fontsize = 20


ax_cc_reg_hum = figure.add_axes([-0.145, 0.08, 0.88, 0.88])
ax_cc_reg_ml = figure.add_axes([0.32, 0.08, 0.88, 0.88])

helper_func.plot_reg_map(ax=ax_cc_reg_hum, gauss_map=gauss_dict_ubvi_hum_12['gauss_map'], seg_map=gauss_dict_ubvi_hum_12['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)

helper_func.plot_reg_map(ax=ax_cc_reg_ml, gauss_map=gauss_dict_ubvi_ml_12['gauss_map'], seg_map=gauss_dict_ubvi_ml_12['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)

ax_cc_reg_hum.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-', label=r'BC03, Z$_{\odot}$')
ax_cc_reg_hum.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}/50$')
ax_cc_reg_ml.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
ax_cc_reg_ml.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')


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

ax_cc_reg_hum.plot(vi_hull_young_hum_12, ub_hull_young_hum_12, color='tab:blue')
ax_cc_reg_hum.plot(vi_hull_cascade_hum_12, ub_hull_cascade_hum_12, color='tab:green')
ax_cc_reg_hum.plot(vi_hull_gc_hum_12, ub_hull_gc_hum_12, color='tab:red')

ax_cc_reg_ml.plot(vi_hull_young_ml_12, ub_hull_young_ml_12, color='tab:blue')
ax_cc_reg_ml.plot(vi_hull_cascade_ml_12, ub_hull_cascade_ml_12, color='tab:green')
ax_cc_reg_ml.plot(vi_hull_gc_ml_12, ub_hull_gc_ml_12, color='tab:red')



vi_int = 1.0
ub_int = -1.3
av_value = 1

helper_func.plot_reddening_vect(ax=ax_cc_reg_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='tab:red', text=True, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax_cc_reg_ml, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.05, y_text_offset=-0.05,
                       linewidth=4, line_color='tab:red', text=True, fontsize=fontsize)

ax_cc_reg_hum.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_ml.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

# ax_cc_reg_hum.set_title('Uncertainty weighted density', fontsize=fontsize)

ax_cc_reg_hum.set_xlim(x_lim_vi)
ax_cc_reg_ml.set_xlim(x_lim_vi)

ax_cc_reg_hum.set_ylim(y_lim_ub)
ax_cc_reg_ml.set_ylim(y_lim_ub)

# ax_cc_reg_hum.set_yticklabels([])
ax_cc_reg_ml.set_yticklabels([])

# ax_cc_reg_hum.set_xticklabels([])

ax_cc_reg_hum.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_cc_reg_hum.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_reg_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_cc_reg_hum.legend(frameon=False, loc=3, fontsize=fontsize)

ax_cc_reg_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_reg_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


# plt.show()
# plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('plot_output/uncert_reg_c12_ellipse.png')
plt.savefig('plot_output/uncert_reg_c12_ellipse.pdf')







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
