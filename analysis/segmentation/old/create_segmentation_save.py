import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import dust_tools.extinction_tools
from photometry_tools import helper_func

import sep
from scipy.spatial import ConvexHull


def fit_and_plot_line(ax, x_data, y_data, x_data_err, y_data_err):
    lin_fit_result = helper_func.fit_line(x_data=x_data, y_data=y_data, x_data_err=x_data_err, y_data_err=y_data_err)

    dummy_x_data = np.linspace(x_lim_vi[0], x_lim_vi[1], 100)
    dummy_y_data = helper_func.lin_func((lin_fit_result['gradient'], lin_fit_result['intersect']), x=dummy_x_data)
    ax.plot(dummy_x_data, dummy_y_data, color='k', linewidth=2, linestyle='--')

    x_text_pos = 0.8

    text_anle = np.arctan(lin_fit_result['gradient']) * 180/np.pi

    ax.text(x_text_pos-0.12,
                  helper_func.lin_func((lin_fit_result['gradient'], lin_fit_result['intersect']),
                           x=x_text_pos)+0.05,
                  r'slope = %.2f$\,\pm\,$%.2f' %
                  (lin_fit_result['gradient'], lin_fit_result['gradient_err']),
                  horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
                  rotation=text_anle, fontsize=fontsize - 5)



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
model_nuvb_sol = np.load('../color_color/data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_nuvb_sol50 = np.load('../color_color/data_output/model_nuvb_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')
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

# color range limitations
mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_detect_nuvbvi_hum = detect_nuv_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_bvvi_hum = detect_b_hum * detect_v_hum * detect_i_hum

mask_detect_nuvbvi_ml = detect_nuv_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_detect_bvvi_ml = detect_b_ml * detect_v_ml * detect_i_ml

x_lim_vi = (-0.7, 1.7)
y_lim_nuvb = (2.4, -2.9)
y_lim_ub = (1.3, -2.2)

# get gauss und segmentations
n_bins_nuvbvi = 100
n_bins_ubvi = 100
n_bins_bvvi = 100
threshold_fact = 2.5
kernal_std = 3
contrast = 0.000001

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


gauss_dict_nuvbvi_hum_1 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                                             y_data=color_nuvb_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                                             y_data_err=color_nuvb_err_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                             threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)
gauss_dict_nuvbvi_hum_2 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                                             y_data=color_nuvb_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                                             y_data_err=color_nuvb_err_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                             threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)
gauss_dict_nuvbvi_hum_3 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                                             y_data=color_nuvb_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                                             y_data_err=color_nuvb_err_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                             threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)

gauss_dict_ubvi_hum_1 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                             y_data=color_ub_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                             y_data_err=color_ub_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_hum_2 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             y_data=color_ub_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             y_data_err=color_ub_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_hum_3 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             y_data=color_ub_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             y_data_err=color_ub_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)


gauss_dict_nuvbvi_ml_1 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                                              y_data=color_nuvb_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                                              y_data_err=color_nuvb_err_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                              threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)
gauss_dict_nuvbvi_ml_2 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                                              y_data=color_nuvb_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                                              y_data_err=color_nuvb_err_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                              threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)
gauss_dict_nuvbvi_ml_3 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                                              y_data=color_nuvb_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                                              y_data_err=color_nuvb_err_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                              threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)

gauss_dict_ubvi_ml_1 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                            y_data=color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                            x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                            y_data_err=color_ub_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                            x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                            threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_ml_2 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                            y_data=color_ub_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                            x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                            y_data_err=color_ub_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                            x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                            threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_ml_3 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                            y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                            x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                            y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                            x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                            threshold_fact=threshold_fact, kernal_std=kernal_std, contrast=contrast)


vi_hull_young_nuvbvi_hum_2, nuvb_hull_young_nuvbvi_hum_2 = helper_func.seg2hull(gauss_dict_nuvbvi_hum_2['seg_deb_map'],
                                                                                x_lim=x_lim_vi, y_lim=y_lim_nuvb,
                                                                                n_bins=n_bins_nuvbvi,
                                                                                seg_index=1, contour_index=0,
                                                                                save_str='nuvbvi_young_hum_2',
                                                                                x_label='vi', y_label='nuvb')
vi_hull_young_nuvbvi_hum_3, nuvb_hull_young_nuvbvi_hum_3 = helper_func.seg2hull(gauss_dict_nuvbvi_hum_3['seg_deb_map'],
                                                                                x_lim=x_lim_vi, y_lim=y_lim_nuvb,
                                                                                n_bins=n_bins_nuvbvi,
                                                                                seg_index=1, contour_index=0,
                                                                                save_str='nuvbvi_young_hum_2',
                                                                                x_label='vi', y_label='nuvb')
vi_hull_young_nuvbvi_ml_2, nuvb_hull_young_nuvbvi_ml_2 = helper_func.seg2hull(gauss_dict_nuvbvi_ml_2['seg_deb_map'],
                                                                                x_lim=x_lim_vi, y_lim=y_lim_nuvb,
                                                                                n_bins=n_bins_nuvbvi,
                                                                                seg_index=1, contour_index=0,
                                                                                save_str='nuvbvi_young_ml_2',
                                                                                x_label='vi', y_label='nuvb')
vi_hull_young_nuvbvi_ml_3, nuvb_hull_young_nuvbvi_ml_3 = helper_func.seg2hull(gauss_dict_nuvbvi_ml_3['seg_deb_map'],
                                                                                x_lim=x_lim_vi, y_lim=y_lim_nuvb,
                                                                                n_bins=n_bins_nuvbvi,
                                                                                seg_index=1, contour_index=0,
                                                                                save_str='nuvbvi_young_ml_2',
                                                                                x_label='vi', y_label='nuvb')

vi_hull_young_ubvi_hum_2, ub_hull_young_ubvi_hum_2 = helper_func.seg2hull(gauss_dict_ubvi_hum_2['seg_deb_map'],
                                                                                x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                                n_bins=n_bins_ubvi,
                                                                                seg_index=1, contour_index=0,
                                                                                save_str='ubvi_young_hum_2',
                                                                                x_label='vi', y_label='ub')
vi_hull_young_ubvi_hum_3, ub_hull_young_ubvi_hum_3 = helper_func.seg2hull(gauss_dict_ubvi_hum_3['seg_deb_map'],
                                                                                x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                                n_bins=n_bins_ubvi,
                                                                                seg_index=1, contour_index=0,
                                                                                save_str='ubvi_young_hum_2',
                                                                                x_label='vi', y_label='ub')
vi_hull_young_ubvi_ml_2, ub_hull_young_ubvi_ml_2 = helper_func.seg2hull(gauss_dict_ubvi_ml_2['seg_deb_map'],
                                                                                x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                                n_bins=n_bins_ubvi,
                                                                                seg_index=1, contour_index=0,
                                                                                save_str='ubvi_young_ml_2',
                                                                                x_label='vi', y_label='ub')
vi_hull_young_ubvi_ml_3, ub_hull_young_ubvi_ml_3 = helper_func.seg2hull(gauss_dict_ubvi_ml_3['seg_deb_map'],
                                                                                x_lim=x_lim_vi, y_lim=y_lim_ub,
                                                                                n_bins=n_bins_ubvi,
                                                                                seg_index=1, contour_index=0,
                                                                                save_str='ubvi_young_ml_2',
                                                                                x_label='vi', y_label='ub')

hull_nuvbvi_young_hum_2 = ConvexHull(np.array([vi_hull_young_nuvbvi_hum_2, nuvb_hull_young_nuvbvi_hum_2]).T)
hull_nuvbvi_young_hum_3 = ConvexHull(np.array([vi_hull_young_nuvbvi_hum_3, nuvb_hull_young_nuvbvi_hum_3]).T)
hull_nuvbvi_young_ml_2 = ConvexHull(np.array([vi_hull_young_nuvbvi_ml_2, nuvb_hull_young_nuvbvi_ml_2]).T)
hull_nuvbvi_young_ml_3 = ConvexHull(np.array([vi_hull_young_nuvbvi_ml_3, nuvb_hull_young_nuvbvi_ml_3]).T)

hull_ubvi_young_hum_2 = ConvexHull(np.array([vi_hull_young_ubvi_hum_2, ub_hull_young_ubvi_hum_2]).T)
hull_ubvi_young_hum_3 = ConvexHull(np.array([vi_hull_young_ubvi_hum_3, ub_hull_young_ubvi_hum_3]).T)
hull_ubvi_young_ml_2 = ConvexHull(np.array([vi_hull_young_ubvi_ml_2, ub_hull_young_ubvi_ml_2]).T)
hull_ubvi_young_ml_3 = ConvexHull(np.array([vi_hull_young_ubvi_ml_3, ub_hull_young_ubvi_ml_3]).T)

in_hull_young_nuvbvi_hum_2 = helper_func.points_in_hull(np.array([color_vi_hum, color_nuvb_hum]).T, hull_nuvbvi_young_hum_2)
in_hull_young_nuvbvi_hum_3 = helper_func.points_in_hull(np.array([color_vi_hum, color_nuvb_hum]).T, hull_nuvbvi_young_hum_3)
in_hull_young_nuvbvi_ml_2 = helper_func.points_in_hull(np.array([color_vi_ml, color_nuvb_ml]).T, hull_nuvbvi_young_ml_2)
in_hull_young_nuvbvi_ml_3 = helper_func.points_in_hull(np.array([color_vi_ml, color_nuvb_ml]).T, hull_nuvbvi_young_ml_3)
in_hull_young_ubvi_hum_2 = helper_func.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_ubvi_young_hum_2)
in_hull_young_ubvi_hum_3 = helper_func.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_ubvi_young_hum_3)
in_hull_young_ubvi_ml_2 = helper_func.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_ubvi_young_ml_2)
in_hull_young_ubvi_ml_3 = helper_func.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_ubvi_young_ml_3)


gauss_map_hum_2 = gauss_dict_nuvbvi_hum_2['gauss_map']
seg_map_hum_2 = gauss_dict_nuvbvi_hum_2['seg_deb_map']
young_map_hum_2 = gauss_map_hum_2.copy()
cascade_map_hum_2 = gauss_map_hum_2.copy()
young_map_hum_2[seg_map_hum_2._data != 1] = 0
cascade_map_hum_2[seg_map_hum_2._data != 2] = 0

sep_table_young_hum_2 = sep.extract(data=young_map_hum_2, thresh=np.nanmax(young_map_hum_2)/1000)
frac_young_hum_2, scale_young_hum_2 = get_scale(
    x_points=color_vi_hum[in_hull_young_nuvbvi_hum_2 * mask_class_2_hum * mask_good_colors_nuvbvi_hum],
    y_points=color_nuvb_hum[in_hull_young_nuvbvi_hum_2 * mask_class_2_hum * mask_good_colors_nuvbvi_hum],
    x_lim=x_lim_vi, y_lim=y_lim_nuvb, map_shape=young_map_hum_2.shape, sep_table=sep_table_young_hum_2,
    table_index=0, n_frac=0.68, max_counts=10000)


fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(30, 40))
fontsize = 35


helper_func.plot_reg_map(ax=ax[0, 0], gauss_map=gauss_dict_nuvbvi_hum_1['gauss_map'],
                         seg_map=gauss_dict_nuvbvi_hum_1['seg_deb_map'], smooth_kernel=kernal_std,
                         color_1='Greens', color_2='Reds',
                         x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi)
helper_func.plot_reg_map(ax=ax[0, 1], gauss_map=gauss_dict_nuvbvi_hum_2['gauss_map'],
                         seg_map=gauss_dict_nuvbvi_hum_2['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi)
helper_func.plot_reg_map(ax=ax[0, 2], gauss_map=gauss_dict_nuvbvi_hum_3['gauss_map'],
                         seg_map=gauss_dict_nuvbvi_hum_3['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi)
helper_func.plot_reg_map(ax=ax[1, 0], gauss_map=gauss_dict_nuvbvi_ml_1['gauss_map'],
                         seg_map=gauss_dict_nuvbvi_ml_1['seg_deb_map'], smooth_kernel=kernal_std,
                         color_1='Greens', color_2='Reds',
                         x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi)
helper_func.plot_reg_map(ax=ax[1, 1], gauss_map=gauss_dict_nuvbvi_ml_2['gauss_map'],
                         seg_map=gauss_dict_nuvbvi_ml_2['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi)
helper_func.plot_reg_map(ax=ax[1, 2], gauss_map=gauss_dict_nuvbvi_ml_3['gauss_map'],
                         seg_map=gauss_dict_nuvbvi_ml_3['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi)
helper_func.plot_reg_map(ax=ax[2, 0], gauss_map=gauss_dict_ubvi_hum_1['gauss_map'],
                         seg_map=gauss_dict_ubvi_hum_1['seg_deb_map'], smooth_kernel=kernal_std,
                         color_1='Greens', color_2='Reds',
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)
helper_func.plot_reg_map(ax=ax[2, 1], gauss_map=gauss_dict_ubvi_hum_2['gauss_map'],
                         seg_map=gauss_dict_ubvi_hum_2['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)
helper_func.plot_reg_map(ax=ax[2, 2], gauss_map=gauss_dict_ubvi_hum_3['gauss_map'],
                         seg_map=gauss_dict_ubvi_hum_3['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)
helper_func.plot_reg_map(ax=ax[3, 0], gauss_map=gauss_dict_ubvi_ml_1['gauss_map'],
                         seg_map=gauss_dict_ubvi_ml_1['seg_deb_map'], smooth_kernel=kernal_std,
                         color_1='Greens', color_2='Reds',
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)
helper_func.plot_reg_map(ax=ax[3, 1], gauss_map=gauss_dict_ubvi_ml_2['gauss_map'],
                         seg_map=gauss_dict_ubvi_ml_2['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)
helper_func.plot_reg_map(ax=ax[3, 2], gauss_map=gauss_dict_ubvi_ml_3['gauss_map'],
                         seg_map=gauss_dict_ubvi_ml_3['seg_deb_map'], smooth_kernel=kernal_std,
                         x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi)

plot_rescaled_sep_ellipse(ax=ax[0, 1], sep_table=sep_table_young_hum_2, map_shape=young_map_hum_2.shape,
                          x_lim=x_lim_vi, y_lim=y_lim_nuvb, scale=scale_young_hum_2, table_index=0, color='k')
ax[0, 1].scatter(sep_table_young_hum_2['x'][0] / young_map_hum_2.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0],
                      sep_table_young_hum_2['y'][0] / young_map_hum_2.shape[1] * (y_lim_nuvb[0] - y_lim_nuvb[1]) + y_lim_nuvb[1],
                      color='k', edgecolor='k', s=80)


fit_and_plot_line(ax=ax[0, 1],
                  x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum_2],
                  y_data=color_nuvb_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum_2],
                  x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum_2],
                  y_data_err=color_nuvb_err_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum_2])
fit_and_plot_line(ax=ax[0, 2],
                  x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum_3],
                  y_data=color_nuvb_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum_3],
                  x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum_3],
                  y_data_err=color_nuvb_err_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum_3])
fit_and_plot_line(ax=ax[1, 1],
                  x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml_2],
                  y_data=color_nuvb_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml_2],
                  x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml_2],
                  y_data_err=color_nuvb_err_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml_2])
fit_and_plot_line(ax=ax[1, 2],
                  x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml_3],
                  y_data=color_nuvb_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml_3],
                  x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml_3],
                  y_data_err=color_nuvb_err_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml_3])
fit_and_plot_line(ax=ax[2, 1],
                  x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum_2],
                  y_data=color_ub_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum_2],
                  x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum_2],
                  y_data_err=color_ub_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum_2])
fit_and_plot_line(ax=ax[2, 2],
                  x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum_3],
                  y_data=color_ub_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum_3],
                  x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum_3],
                  y_data_err=color_ub_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum_3])
fit_and_plot_line(ax=ax[3, 1],
                  x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml_2],
                  y_data=color_ub_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml_2],
                  x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml_2],
                  y_data_err=color_ub_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml_2])
fit_and_plot_line(ax=ax[3, 2],
                  x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml_3],
                  y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml_3],
                  x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml_3],
                  y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml_3])



vi_int = 1.0
nuvb_int = -2.2
ub_int = -1.8
av_value = 1

catalog_access = photometry_tools.data_access.CatalogAccess()
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
nuv_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F275W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=av_value)
max_color_ext_nuvb = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=nuv_wave, wave2=b_wave, av=av_value)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=av_value)

slope_av_vector_nuvbvi = ((nuvb_int + max_color_ext_nuvb) - nuvb_int) / ((vi_int + max_color_ext_vi) - vi_int)
angle_av_vector_nuvbvi = np.arctan(slope_av_vector_nuvbvi) * 180/np.pi
slope_av_vector_ubvi = ((ub_int + max_color_ext_ub) - ub_int) / ((vi_int + max_color_ext_vi) - vi_int)
angle_av_vector_ubvi = np.arctan(slope_av_vector_ubvi) * 180/np.pi

print('slope_av_vector_nuvbvi ', slope_av_vector_nuvbvi)
print('angle_av_vector_nuvbvi ', angle_av_vector_nuvbvi)
print('slope_av_vector_ubvi ', slope_av_vector_ubvi)
print('angle_av_vector_ubvi ', angle_av_vector_ubvi)


ax[0, 1].text(vi_int-0.12, nuvb_int+0.05,
              r'slope = %.2f' % (slope_av_vector_nuvbvi),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=angle_av_vector_nuvbvi, fontsize=fontsize - 5)
ax[2, 1].text(vi_int-0.12, ub_int+0.05,
              r'slope = %.2f' % (slope_av_vector_ubvi),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=angle_av_vector_ubvi, fontsize=fontsize - 5)







helper_func.plot_reddening_vect(ax=ax[0, 0], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
helper_func.plot_reddening_vect(ax=ax[0, 1], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[0, 2], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[1, 0], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
helper_func.plot_reddening_vect(ax=ax[1, 1], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[1, 2], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[2, 0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[2, 1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[2, 2], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[3, 0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[3, 1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[3, 2], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


helper_func.display_models(ax=ax[0, 0], x_color_sol=model_vi_sol, y_color_sol=model_nuvb_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_nuvb_sol50, age_sol50=age_mod_sol50,
                           age_label_fontsize=fontsize+2, label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
helper_func.display_models(ax=ax[0, 1], x_color_sol=model_vi_sol, y_color_sol=model_nuvb_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_nuvb_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[0, 2], x_color_sol=model_vi_sol, y_color_sol=model_nuvb_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_nuvb_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[1, 0], x_color_sol=model_vi_sol, y_color_sol=model_nuvb_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_nuvb_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[1, 1], x_color_sol=model_vi_sol, y_color_sol=model_nuvb_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_nuvb_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[1, 2], x_color_sol=model_vi_sol, y_color_sol=model_nuvb_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_nuvb_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[2, 0], x_color_sol=model_vi_sol, y_color_sol=model_ub_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_ub_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[2, 1], x_color_sol=model_vi_sol, y_color_sol=model_ub_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_ub_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[2, 2], x_color_sol=model_vi_sol, y_color_sol=model_ub_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_ub_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[3, 0], x_color_sol=model_vi_sol, y_color_sol=model_ub_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_ub_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[3, 1], x_color_sol=model_vi_sol, y_color_sol=model_ub_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_ub_sol50, age_sol50=age_mod_sol50,)
helper_func.display_models(ax=ax[3, 2], x_color_sol=model_vi_sol, y_color_sol=model_ub_sol, age_sol=age_mod_sol,
                           x_color_sol50=model_vi_sol50, y_color_sol50=model_ub_sol50, age_sol50=age_mod_sol50,)


ax[0, 0].set_title('Class 1', fontsize=fontsize)
ax[0, 1].set_title('Class 2', fontsize=fontsize)
ax[0, 2].set_title('Compact Association', fontsize=fontsize)

ax[3, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[3, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[3, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax[0, 0].set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')',labelpad=30, fontsize=fontsize)
ax[1, 0].set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')',labelpad=30, fontsize=fontsize)
ax[2, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')',labelpad=0, fontsize=fontsize)
ax[3, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')',labelpad=0, fontsize=fontsize)


ax[0, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.23,
              'HUM', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.15,
              'HUM', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.05,
              'HUM', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.05,
              'ML', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.05,
              'ML', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.05,
              'ML', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[2, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'HUM', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[2, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'HUM', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[2, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'HUM', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[3, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'ML', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[3, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'ML', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[3, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'ML', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

xedges = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_nuvbvi)
yedges = np.linspace(y_lim_nuvb[0], y_lim_nuvb[1], n_bins_nuvbvi)
kernal_rad_width = (xedges[1] - xedges[0]) * kernal_std
kernal_rad_hight = (yedges[1] - yedges[0]) * kernal_std
# plot_kernel_std
ellipse = Ellipse(xy=(-0.5, 2.0), width=kernal_rad_width, height=kernal_rad_hight, angle=0, edgecolor='r', fc='None', lw=2)
ax[0, 1].add_patch(ellipse)
ax[0, 1].text(-0.4, 2.0, 'Smoothing', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[0, 0].legend(frameon=False, loc=3, #bbox_to_anchor=(0, 0.05),
                fontsize=fontsize-2)


ax[0, 0].set_xticks([])
ax[0, 1].set_xticks([])
ax[0, 2].set_xticks([])
ax[1, 0].set_xticks([])
ax[1, 1].set_xticks([])
ax[1, 2].set_xticks([])
ax[2, 0].set_xticks([])
ax[2, 1].set_xticks([])
ax[2, 2].set_xticks([])


ax[0, 1].set_yticks([])
ax[0, 2].set_yticks([])
ax[1, 1].set_yticks([])
ax[1, 2].set_yticks([])
ax[2, 1].set_yticks([])
ax[2, 2].set_yticks([])
ax[3, 1].set_yticks([])
ax[3, 2].set_yticks([])


ax[0, 0].set_xlim(x_lim_vi)
ax[0, 1].set_xlim(x_lim_vi)
ax[0, 2].set_xlim(x_lim_vi)
ax[1, 0].set_xlim(x_lim_vi)
ax[1, 1].set_xlim(x_lim_vi)
ax[1, 2].set_xlim(x_lim_vi)
ax[2, 0].set_xlim(x_lim_vi)
ax[2, 1].set_xlim(x_lim_vi)
ax[2, 2].set_xlim(x_lim_vi)
ax[3, 0].set_xlim(x_lim_vi)
ax[3, 1].set_xlim(x_lim_vi)
ax[3, 2].set_xlim(x_lim_vi)

ax[0, 0].set_ylim(y_lim_nuvb)
ax[0, 1].set_ylim(y_lim_nuvb)
ax[0, 2].set_ylim(y_lim_nuvb)
ax[1, 0].set_ylim(y_lim_nuvb)
ax[1, 1].set_ylim(y_lim_nuvb)
ax[1, 2].set_ylim(y_lim_nuvb)
ax[2, 0].set_ylim(y_lim_ub)
ax[2, 1].set_ylim(y_lim_ub)
ax[2, 2].set_ylim(y_lim_ub)
ax[3, 0].set_ylim(y_lim_ub)
ax[3, 1].set_ylim(y_lim_ub)
ax[3, 2].set_ylim(y_lim_ub)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[3, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[3, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[3, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


fig.subplots_adjust(left=0.06, bottom=0.04, right=0.995, top=0.98, wspace=0.01, hspace=0.01)
plt.savefig('plot_output/segment_maps.pdf')
plt.savefig('plot_output/segment_maps.png')




