import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import dust_tools.extinction_tools
from photometry_tools import helper_func

from astropy.io import fits
from scipy.spatial import ConvexHull



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

            ax.annotate(' ', #annotation_dict[age]['label'],
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



def fit_and_plot_line(ax, x_data, y_data, x_data_err, y_data_err):
    lin_fit_result = helper_func.fit_line(x_data=x_data, y_data=y_data,
                                                      x_data_err=x_data_err, y_data_err=y_data_err)

    dummy_x_data = np.linspace(x_lim_vi[0], x_lim_vi[1], 100)
    dummy_y_data = helper_func.lin_func((lin_fit_result['gradient'], lin_fit_result['intersect']), x=dummy_x_data)
    ax.plot(dummy_x_data, dummy_y_data, color='k', linewidth=2, linestyle='--')

    x_text_pos = 1.0

    text_anle = np.arctan(lin_fit_result['gradient']) * 180/np.pi

    ax.text(x_text_pos-0.12,
                  helper_func.lin_func((lin_fit_result['gradient'], lin_fit_result['intersect']),
                           x=x_text_pos)+0.05,
                  r'slope = %.2f$\,\pm\,$%.2f' %
                  (lin_fit_result['gradient'], lin_fit_result['gradient_err']),
                  horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
                  rotation=text_anle, fontsize=fontsize - 5)





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

x_lim_vi = (-0.7, 2.4)

y_lim_nuvb = (3.2, -2.9)
y_lim_ub = (2.1, -2.2)
y_lim_bv = (1.9, -0.7)
n_bins = 190
kernal_std = 3.0

mask_good_colors_nuvbvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_nuvb_hum > (y_lim_nuvb[1] - 1)) & (color_nuvb_hum < (y_lim_nuvb[0] + 1)) &
                               mask_detect_nuvbvi_hum)
mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_hum)
mask_good_colors_bvvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_bv_hum > (y_lim_bv[1] - 1)) & (color_bv_hum < (y_lim_bv[0] + 1)) &
                               mask_detect_bvvi_hum)

mask_good_colors_nuvbvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_nuvb_ml > (y_lim_nuvb[1] - 1)) & (color_nuvb_ml < (y_lim_nuvb[0] + 1)) &
                               mask_detect_nuvbvi_ml)
mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_ml)
mask_good_colors_bvvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_bv_ml > (y_lim_bv[1] - 1)) & (color_bv_ml < (y_lim_bv[0] + 1)) &
                               mask_detect_bvvi_ml)


convex_hull_young_nuvbvi_hum = fits.open('../color_color_regions/data_output/convex_hull_nuvbvi_young_hum_12.fits')[1].data
convex_hull_young_nuvbvi_ml = fits.open('../color_color_regions/data_output/convex_hull_nuvbvi_young_ml_12.fits')[1].data

convex_hull_young_ubvi_hum = fits.open('../color_color_regions/data_output/convex_hull_ubvi_young_hum_12.fits')[1].data
convex_hull_young_ubvi_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_young_ml_12.fits')[1].data

convex_hull_young_bvvi_hum = fits.open('../color_color_regions/data_output/convex_hull_bvvi_young_hum_12.fits')[1].data
convex_hull_young_bvvi_ml = fits.open('../color_color_regions/data_output/convex_hull_bvvi_young_ml_12.fits')[1].data

convex_hull_nuvbvi_vi_young_hum = convex_hull_young_nuvbvi_hum['vi']
convex_hull_nuvbvi_nuvb_young_hum = convex_hull_young_nuvbvi_hum['nuvb']
convex_hull_nuvbvi_vi_young_ml = convex_hull_young_nuvbvi_ml['vi']
convex_hull_nuvbvi_nuvb_young_ml = convex_hull_young_nuvbvi_ml['nuvb']

convex_hull_ubvi_vi_young_hum = convex_hull_young_ubvi_hum['vi']
convex_hull_ubvi_ub_young_hum = convex_hull_young_ubvi_hum['ub']
convex_hull_ubvi_vi_young_ml = convex_hull_young_ubvi_ml['vi']
convex_hull_ubvi_ub_young_ml = convex_hull_young_ubvi_ml['ub']

convex_hull_bvvi_vi_young_hum = convex_hull_young_bvvi_hum['vi']
convex_hull_bvvi_bv_young_hum = convex_hull_young_bvvi_hum['bv']
convex_hull_bvvi_vi_young_ml = convex_hull_young_bvvi_ml['vi']
convex_hull_bvvi_bv_young_ml = convex_hull_young_bvvi_ml['bv']

hull_young_nuvbvi_hum = ConvexHull(np.array([convex_hull_nuvbvi_vi_young_hum, convex_hull_nuvbvi_nuvb_young_hum]).T)
hull_young_nuvbvi_ml = ConvexHull(np.array([convex_hull_nuvbvi_vi_young_ml, convex_hull_nuvbvi_nuvb_young_ml]).T)
in_hull_young_nuvbvi_hum = helper_func.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_young_nuvbvi_hum)
in_hull_young_nuvbvi_ml = helper_func.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_young_nuvbvi_ml)

hull_young_ubvi_hum = ConvexHull(np.array([convex_hull_ubvi_vi_young_hum, convex_hull_ubvi_ub_young_hum]).T)
hull_young_ubvi_ml = ConvexHull(np.array([convex_hull_ubvi_vi_young_ml, convex_hull_ubvi_ub_young_ml]).T)
in_hull_young_ubvi_hum = helper_func.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_young_ubvi_hum)
in_hull_young_ubvi_ml = helper_func.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_young_ubvi_ml)

hull_young_bvvi_hum = ConvexHull(np.array([convex_hull_bvvi_vi_young_hum, convex_hull_bvvi_bv_young_hum]).T)
hull_young_bvvi_ml = ConvexHull(np.array([convex_hull_bvvi_vi_young_ml, convex_hull_bvvi_bv_young_ml]).T)
in_hull_young_bvvi_hum = helper_func.points_in_hull(np.array([color_vi_hum, color_bv_hum]).T, hull_young_bvvi_hum)
in_hull_young_bvvi_ml = helper_func.points_in_hull(np.array([color_vi_ml, color_bv_ml]).T, hull_young_bvvi_ml)

# get gauss und segmentations
n_bins_nuvbvi = 120
n_bins_ubvi = 120
n_bins_bvvi = 120
threshold_fact = 3
kernal_std = 1.0
contrast = 0.01

gauss_dict_nuvbvi_hum_1 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                                             y_data=color_nuvb_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                                             y_data_err=color_nuvb_err_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_nuvbvi_hum_2 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                                             y_data=color_nuvb_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                                             y_data_err=color_nuvb_err_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_nuvbvi_hum_3 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                                             y_data=color_nuvb_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                                             y_data_err=color_nuvb_err_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

gauss_dict_ubvi_hum_1 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                             y_data=color_ub_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                             y_data_err=color_ub_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_hum_2 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             y_data=color_ub_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             y_data_err=color_ub_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_hum_3 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             y_data=color_ub_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             y_data_err=color_ub_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

gauss_dict_bvvi_hum_1 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                             y_data=color_bv_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                             y_data_err=color_bv_err_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_bvvi_hum_2 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_bvvi_hum],
                                             y_data=color_bv_hum[mask_class_2_hum * mask_good_colors_bvvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_bvvi_hum],
                                             y_data_err=color_bv_err_hum[mask_class_2_hum * mask_good_colors_bvvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_bvvi_hum_3 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_bvvi_hum],
                                             y_data=color_bv_hum[mask_class_3_hum * mask_good_colors_bvvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_bvvi_hum],
                                             y_data_err=color_bv_err_hum[mask_class_3_hum * mask_good_colors_bvvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)


gauss_dict_nuvbvi_ml_1 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                                              y_data=color_nuvb_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                                              y_data_err=color_nuvb_err_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_nuvbvi_ml_2 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                                              y_data=color_nuvb_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                                              y_data_err=color_nuvb_err_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_nuvbvi_ml_3 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                                              y_data=color_nuvb_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                                              y_data_err=color_nuvb_err_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins_nuvbvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)


gauss_dict_ubvi_ml_1 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                              y_data=color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                              y_data_err=color_ub_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_ml_2 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                              y_data=color_ub_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                              y_data_err=color_ub_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_ml_3 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

gauss_dict_bvvi_ml_1 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                              y_data=color_bv_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                              y_data_err=color_bv_err_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_bvvi_ml_2 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_bvvi_ml],
                                              y_data=color_bv_ml[mask_class_2_ml * mask_good_colors_bvvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_bvvi_ml],
                                              y_data_err=color_bv_err_ml[mask_class_2_ml * mask_good_colors_bvvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_bvvi_ml_3 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_bvvi_ml],
                                              y_data=color_bv_ml[mask_class_3_ml * mask_good_colors_bvvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_bvvi_ml],
                                              y_data_err=color_bv_err_ml[mask_class_3_ml * mask_good_colors_bvvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)


x_bins_vi = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_ubvi)
y_bins_nuvb = np.linspace(y_lim_nuvb[1], y_lim_nuvb[0], n_bins_nuvbvi)
y_bins_ub = np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins_ubvi)
y_bins_bv = np.linspace(y_lim_bv[1], y_lim_bv[0], n_bins_bvvi)
x_mesh_vi, y_mesh_nuvb = np.meshgrid(x_bins_vi, y_bins_nuvb)
x_mesh_vi, y_mesh_ub = np.meshgrid(x_bins_vi, y_bins_ub)
x_mesh_vi, y_mesh_bv = np.meshgrid(x_bins_vi, y_bins_bv)

gauss_map_nuvbvi_hum_1_bkg = gauss_dict_nuvbvi_hum_1['gauss_map'].copy()
gauss_map_nuvbvi_hum_1_young = gauss_dict_nuvbvi_hum_1['gauss_map'].copy()
gauss_map_nuvbvi_hum_2_bkg = gauss_dict_nuvbvi_hum_2['gauss_map'].copy()
gauss_map_nuvbvi_hum_2_young = gauss_dict_nuvbvi_hum_2['gauss_map'].copy()
gauss_map_nuvbvi_hum_3_bkg = gauss_dict_nuvbvi_hum_3['gauss_map'].copy()
gauss_map_nuvbvi_hum_3_young = gauss_dict_nuvbvi_hum_3['gauss_map'].copy()

gauss_map_ubvi_hum_1_bkg = gauss_dict_ubvi_hum_1['gauss_map'].copy()
gauss_map_ubvi_hum_1_young = gauss_dict_ubvi_hum_1['gauss_map'].copy()
gauss_map_ubvi_hum_2_bkg = gauss_dict_ubvi_hum_2['gauss_map'].copy()
gauss_map_ubvi_hum_2_young = gauss_dict_ubvi_hum_2['gauss_map'].copy()
gauss_map_ubvi_hum_3_bkg = gauss_dict_ubvi_hum_3['gauss_map'].copy()
gauss_map_ubvi_hum_3_young = gauss_dict_ubvi_hum_3['gauss_map'].copy()

gauss_map_bvvi_hum_1_bkg = gauss_dict_bvvi_hum_1['gauss_map'].copy()
gauss_map_bvvi_hum_1_young = gauss_dict_bvvi_hum_1['gauss_map'].copy()
gauss_map_bvvi_hum_2_bkg = gauss_dict_bvvi_hum_2['gauss_map'].copy()
gauss_map_bvvi_hum_2_young = gauss_dict_bvvi_hum_2['gauss_map'].copy()
gauss_map_bvvi_hum_3_bkg = gauss_dict_bvvi_hum_3['gauss_map'].copy()
gauss_map_bvvi_hum_3_young = gauss_dict_bvvi_hum_3['gauss_map'].copy()


gauss_map_nuvbvi_ml_1_bkg = gauss_dict_nuvbvi_ml_1['gauss_map'].copy()
gauss_map_nuvbvi_ml_1_young = gauss_dict_nuvbvi_ml_1['gauss_map'].copy()
gauss_map_nuvbvi_ml_2_bkg = gauss_dict_nuvbvi_ml_2['gauss_map'].copy()
gauss_map_nuvbvi_ml_2_young = gauss_dict_nuvbvi_ml_2['gauss_map'].copy()
gauss_map_nuvbvi_ml_3_bkg = gauss_dict_nuvbvi_ml_3['gauss_map'].copy()
gauss_map_nuvbvi_ml_3_young = gauss_dict_nuvbvi_ml_3['gauss_map'].copy()

gauss_map_ubvi_ml_1_bkg = gauss_dict_ubvi_ml_1['gauss_map'].copy()
gauss_map_ubvi_ml_1_young = gauss_dict_ubvi_ml_1['gauss_map'].copy()
gauss_map_ubvi_ml_2_bkg = gauss_dict_ubvi_ml_2['gauss_map'].copy()
gauss_map_ubvi_ml_2_young = gauss_dict_ubvi_ml_2['gauss_map'].copy()
gauss_map_ubvi_ml_3_bkg = gauss_dict_ubvi_ml_3['gauss_map'].copy()
gauss_map_ubvi_ml_3_young = gauss_dict_ubvi_ml_3['gauss_map'].copy()

gauss_map_bvvi_ml_1_bkg = gauss_dict_bvvi_ml_1['gauss_map'].copy()
gauss_map_bvvi_ml_1_young = gauss_dict_bvvi_ml_1['gauss_map'].copy()
gauss_map_bvvi_ml_2_bkg = gauss_dict_bvvi_ml_2['gauss_map'].copy()
gauss_map_bvvi_ml_2_young = gauss_dict_bvvi_ml_2['gauss_map'].copy()
gauss_map_bvvi_ml_3_bkg = gauss_dict_bvvi_ml_3['gauss_map'].copy()
gauss_map_bvvi_ml_3_young = gauss_dict_bvvi_ml_3['gauss_map'].copy()

in_hull_map_nuvbvi_hum = np.array(helper_func.points_in_hull(np.array([x_mesh_vi.flatten(), y_mesh_nuvb.flatten()]).T, hull_young_nuvbvi_hum), dtype=bool)
in_hull_map_nuvbvi_ml = np.array(helper_func.points_in_hull(np.array([x_mesh_vi.flatten(), y_mesh_nuvb.flatten()]).T, hull_young_nuvbvi_ml), dtype=bool)
in_hull_map_nuvbvi_hum = np.reshape(in_hull_map_nuvbvi_hum, newshape=(n_bins_nuvbvi, n_bins_nuvbvi))
in_hull_map_nuvbvi_ml = np.reshape(in_hull_map_nuvbvi_ml, newshape=(n_bins_nuvbvi, n_bins_nuvbvi))

in_hull_map_ubvi_hum = np.array(helper_func.points_in_hull(np.array([x_mesh_vi.flatten(), y_mesh_ub.flatten()]).T, hull_young_ubvi_hum), dtype=bool)
in_hull_map_ubvi_ml = np.array(helper_func.points_in_hull(np.array([x_mesh_vi.flatten(), y_mesh_ub.flatten()]).T, hull_young_ubvi_ml), dtype=bool)
in_hull_map_ubvi_hum = np.reshape(in_hull_map_ubvi_hum, newshape=(n_bins_ubvi, n_bins_ubvi))
in_hull_map_ubvi_ml = np.reshape(in_hull_map_ubvi_ml, newshape=(n_bins_ubvi, n_bins_ubvi))

in_hull_map_bvvi_hum = np.array(helper_func.points_in_hull(np.array([x_mesh_vi.flatten(), y_mesh_bv.flatten()]).T, hull_young_bvvi_hum), dtype=bool)
in_hull_map_bvvi_ml = np.array(helper_func.points_in_hull(np.array([x_mesh_vi.flatten(), y_mesh_bv.flatten()]).T, hull_young_bvvi_ml), dtype=bool)
in_hull_map_bvvi_hum = np.reshape(in_hull_map_bvvi_hum, newshape=(n_bins_bvvi, n_bins_bvvi))
in_hull_map_bvvi_ml = np.reshape(in_hull_map_bvvi_ml, newshape=(n_bins_bvvi, n_bins_bvvi))


gauss_map_nuvbvi_hum_2_bkg[in_hull_map_nuvbvi_hum] = np.nan
gauss_map_nuvbvi_hum_2_young[np.invert(in_hull_map_nuvbvi_hum)] = np.nan
gauss_map_nuvbvi_hum_3_bkg[in_hull_map_nuvbvi_hum] = np.nan
gauss_map_nuvbvi_hum_3_young[np.invert(in_hull_map_nuvbvi_hum)] = np.nan

gauss_map_ubvi_hum_2_bkg[in_hull_map_ubvi_hum] = np.nan
gauss_map_ubvi_hum_2_young[np.invert(in_hull_map_ubvi_hum)] = np.nan
gauss_map_ubvi_hum_3_bkg[in_hull_map_ubvi_hum] = np.nan
gauss_map_ubvi_hum_3_young[np.invert(in_hull_map_ubvi_hum)] = np.nan

gauss_map_bvvi_hum_2_bkg[in_hull_map_bvvi_hum] = np.nan
gauss_map_bvvi_hum_2_young[np.invert(in_hull_map_bvvi_hum)] = np.nan
gauss_map_bvvi_hum_3_bkg[in_hull_map_bvvi_hum] = np.nan
gauss_map_bvvi_hum_3_young[np.invert(in_hull_map_bvvi_hum)] = np.nan


gauss_map_nuvbvi_ml_2_bkg[in_hull_map_nuvbvi_ml] = np.nan
gauss_map_nuvbvi_ml_2_young[np.invert(in_hull_map_nuvbvi_ml)] = np.nan
gauss_map_nuvbvi_ml_3_bkg[in_hull_map_nuvbvi_ml] = np.nan
gauss_map_nuvbvi_ml_3_young[np.invert(in_hull_map_nuvbvi_ml)] = np.nan

gauss_map_ubvi_ml_2_bkg[in_hull_map_ubvi_ml] = np.nan
gauss_map_ubvi_ml_2_young[np.invert(in_hull_map_ubvi_ml)] = np.nan
gauss_map_ubvi_ml_3_bkg[in_hull_map_ubvi_ml] = np.nan
gauss_map_ubvi_ml_3_young[np.invert(in_hull_map_ubvi_ml)] = np.nan

gauss_map_bvvi_ml_2_bkg[in_hull_map_bvvi_ml] = np.nan
gauss_map_bvvi_ml_2_young[np.invert(in_hull_map_bvvi_ml)] = np.nan
gauss_map_bvvi_ml_3_bkg[in_hull_map_bvvi_ml] = np.nan
gauss_map_bvvi_ml_3_young[np.invert(in_hull_map_bvvi_ml)] = np.nan


fig, ax = plt.subplots(ncols=3, nrows=9, figsize=(15, 30))
fontsize = 23

vmax_hum_1 = np.nanmax(gauss_dict_nuvbvi_hum_1['gauss_map']) / 1.2
ax[0, 0].imshow(gauss_map_nuvbvi_hum_1_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_1)
ax[0, 0].imshow(gauss_map_nuvbvi_hum_1_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_1)
vmax_hum_2 = np.nanmax(gauss_dict_nuvbvi_hum_2['gauss_map']) / 1.2
ax[0, 1].imshow(gauss_map_nuvbvi_hum_2_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_2)
ax[0, 1].imshow(gauss_map_nuvbvi_hum_2_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_2)
vmax_hum_3 = np.nanmax(gauss_dict_nuvbvi_hum_3['gauss_map']) / 1.2
ax[0, 2].imshow(gauss_map_nuvbvi_hum_3_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_3)
ax[0, 2].imshow(gauss_map_nuvbvi_hum_3_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_3)

vmax_ml_1 = np.nanmax(gauss_dict_nuvbvi_ml_1['gauss_map']) / 1.2
ax[1, 0].imshow(gauss_map_nuvbvi_ml_1_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_1)
ax[1, 0].imshow(gauss_map_nuvbvi_ml_1_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_1)
vmax_ml_2 = np.nanmax(gauss_dict_nuvbvi_ml_2['gauss_map']) / 1.2
ax[1, 1].imshow(gauss_map_nuvbvi_ml_2_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_2)
ax[1, 1].imshow(gauss_map_nuvbvi_ml_2_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_2)
vmax_ml_3 = np.nanmax(gauss_dict_nuvbvi_ml_3['gauss_map']) / 1.2
ax[1, 2].imshow(gauss_map_nuvbvi_ml_3_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_3)
ax[1, 2].imshow(gauss_map_nuvbvi_ml_3_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_3)


vmax_hum_1 = np.nanmax(gauss_dict_ubvi_hum_1['gauss_map']) / 1.2
ax[2, 0].imshow(gauss_map_ubvi_hum_1_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_1)
ax[2, 0].imshow(gauss_map_ubvi_hum_1_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_1)
vmax_hum_2 = np.nanmax(gauss_dict_ubvi_hum_2['gauss_map']) / 1.2
ax[2, 1].imshow(gauss_map_ubvi_hum_2_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_2)
ax[2, 1].imshow(gauss_map_ubvi_hum_2_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_2)
vmax_hum_3 = np.nanmax(gauss_dict_ubvi_hum_3['gauss_map']) / 1.2
ax[2, 2].imshow(gauss_map_ubvi_hum_3_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_3)
ax[2, 2].imshow(gauss_map_ubvi_hum_3_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_3)

vmax_ml_1 = np.nanmax(gauss_dict_ubvi_ml_1['gauss_map']) / 1.2
ax[3, 0].imshow(gauss_map_ubvi_ml_1_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_1)
ax[3, 0].imshow(gauss_map_ubvi_ml_1_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_1)
vmax_ml_2 = np.nanmax(gauss_dict_ubvi_ml_2['gauss_map']) / 1.2
ax[3, 1].imshow(gauss_map_ubvi_ml_2_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_2)
ax[3, 1].imshow(gauss_map_ubvi_ml_2_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_2)
vmax_ml_3 = np.nanmax(gauss_dict_ubvi_ml_3['gauss_map']) / 1.2
ax[3, 2].imshow(gauss_map_ubvi_ml_3_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_3)
ax[3, 2].imshow(gauss_map_ubvi_ml_3_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_3)

vmax_hum_1 = np.nanmax(gauss_dict_bvvi_hum_1['gauss_map']) / 1.2
ax[4, 0].imshow(gauss_map_bvvi_hum_1_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_1)
ax[4, 0].imshow(gauss_map_bvvi_hum_1_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_1)
vmax_hum_2 = np.nanmax(gauss_dict_bvvi_hum_2['gauss_map']) / 1.2
ax[4, 1].imshow(gauss_map_bvvi_hum_2_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_2)
ax[4, 1].imshow(gauss_map_bvvi_hum_2_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_2)
vmax_hum_3 = np.nanmax(gauss_dict_bvvi_hum_3['gauss_map']) / 1.2
ax[4, 2].imshow(gauss_map_bvvi_hum_3_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_3)
ax[4, 2].imshow(gauss_map_bvvi_hum_3_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_3)

vmax_ml_1 = np.nanmax(gauss_dict_bvvi_ml_1['gauss_map']) / 1.2
ax[5, 0].imshow(gauss_map_bvvi_ml_1_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_1)
ax[5, 0].imshow(gauss_map_bvvi_ml_1_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_1)
vmax_ml_2 = np.nanmax(gauss_dict_bvvi_ml_2['gauss_map']) / 1.2
ax[5, 1].imshow(gauss_map_bvvi_ml_2_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_2)
ax[5, 1].imshow(gauss_map_bvvi_ml_2_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_2)
vmax_ml_3 = np.nanmax(gauss_dict_bvvi_ml_3['gauss_map']) / 1.2
ax[5, 2].imshow(gauss_map_bvvi_ml_3_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_3)
ax[5, 2].imshow(gauss_map_bvvi_ml_3_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_3)


fit_and_plot_line(ax=ax[0, 0],
                  x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum],
                  y_data=color_nuvb_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum],
                  x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum],
                  y_data_err=color_nuvb_err_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum])
fit_and_plot_line(ax=ax[0, 1],
                  x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum],
                  y_data=color_nuvb_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum],
                  x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum],
                  y_data_err=color_nuvb_err_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum])
fit_and_plot_line(ax=ax[0, 2],
                  x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum],
                  y_data=color_nuvb_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum],
                  x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum],
                  y_data_err=color_nuvb_err_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum * in_hull_young_nuvbvi_hum])

fit_and_plot_line(ax=ax[1, 0],
                  x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml],
                  y_data=color_nuvb_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml],
                  x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml],
                  y_data_err=color_nuvb_err_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml])
fit_and_plot_line(ax=ax[1, 1],
                  x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml],
                  y_data=color_nuvb_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml],
                  x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml],
                  y_data_err=color_nuvb_err_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml])
fit_and_plot_line(ax=ax[1, 2],
                  x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml],
                  y_data=color_nuvb_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml],
                  x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml],
                  y_data_err=color_nuvb_err_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml * in_hull_young_nuvbvi_ml])


fit_and_plot_line(ax=ax[2, 0],
                  x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum],
                  y_data=color_ub_hum[mask_class_1_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum],
                  x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum],
                  y_data_err=color_ub_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum])
fit_and_plot_line(ax=ax[2, 1],
                  x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum],
                  y_data=color_ub_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum],
                  x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum],
                  y_data_err=color_ub_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum])
fit_and_plot_line(ax=ax[2, 2],
                  x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum],
                  y_data=color_ub_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum],
                  x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum],
                  y_data_err=color_ub_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_ubvi_hum])

fit_and_plot_line(ax=ax[3, 0],
                  x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml],
                  y_data=color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml],
                  x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml],
                  y_data_err=color_ub_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml])
fit_and_plot_line(ax=ax[3, 1],
                  x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml],
                  y_data=color_ub_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml],
                  x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml],
                  y_data_err=color_ub_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml])
fit_and_plot_line(ax=ax[3, 2],
                  x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml],
                  y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml],
                  x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml],
                  y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ubvi_ml])



fit_and_plot_line(ax=ax[4, 0],
                  x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum],
                  y_data=color_bv_hum[mask_class_1_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum],
                  x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum],
                  y_data_err=color_bv_err_hum[mask_class_1_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum])
fit_and_plot_line(ax=ax[4, 1],
                  x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum],
                  y_data=color_bv_hum[mask_class_2_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum],
                  x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum],
                  y_data_err=color_bv_err_hum[mask_class_2_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum])
fit_and_plot_line(ax=ax[4, 2],
                  x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum],
                  y_data=color_bv_hum[mask_class_3_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum],
                  x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum],
                  y_data_err=color_bv_err_hum[mask_class_3_hum * mask_good_colors_bvvi_hum * in_hull_young_bvvi_hum])

fit_and_plot_line(ax=ax[5, 0],
                  x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml],
                  y_data=color_bv_ml[mask_class_1_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml],
                  x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml],
                  y_data_err=color_bv_err_ml[mask_class_1_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml])
fit_and_plot_line(ax=ax[5, 1],
                  x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml],
                  y_data=color_bv_ml[mask_class_2_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml],
                  x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml],
                  y_data_err=color_bv_err_ml[mask_class_2_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml])
fit_and_plot_line(ax=ax[5, 2],
                  x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml],
                  y_data=color_bv_ml[mask_class_3_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml],
                  x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml],
                  y_data_err=color_bv_err_ml[mask_class_3_ml * mask_good_colors_bvvi_ml * in_hull_young_bvvi_ml])


display_models(ax=ax[0, 0], age_label_fontsize=fontsize+2, label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax[0, 1], age_label_fontsize=fontsize+2)
display_models(ax=ax[0, 2], age_label_fontsize=fontsize+2)
display_models(ax=ax[1, 0], age_label_fontsize=fontsize+2, y_color='nuvb')
display_models(ax=ax[1, 1], age_label_fontsize=fontsize+2, y_color='nuvb')
display_models(ax=ax[1, 2], age_label_fontsize=fontsize+2, y_color='nuvb')

display_models(ax=ax[2, 0], age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax[2, 1], age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax[2, 2], age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax[3, 0], age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax[3, 1], age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax[3, 2], age_label_fontsize=fontsize+2, y_color='ub')

display_models(ax=ax[4, 0], age_label_fontsize=fontsize+2, y_color='bv')
display_models(ax=ax[4, 1], age_label_fontsize=fontsize+2, y_color='bv')
display_models(ax=ax[4, 2], age_label_fontsize=fontsize+2, y_color='bv')
display_models(ax=ax[5, 0], age_label_fontsize=fontsize+2, y_color='bv')
display_models(ax=ax[5, 1], age_label_fontsize=fontsize+2, y_color='bv')
display_models(ax=ax[5, 2], age_label_fontsize=fontsize+2, y_color='bv')



vi_int = 1.75
nuvb_int = -2.2
ub_int = -1.8
bv_int = -0.5
av_value = 1

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
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
helper_func.plot_reddening_vect(ax=ax[1, 1], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[1, 2], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

helper_func.plot_reddening_vect(ax=ax[2, 0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
helper_func.plot_reddening_vect(ax=ax[2, 1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[2, 2], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[3, 0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
helper_func.plot_reddening_vect(ax=ax[3, 1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[3, 2], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

helper_func.plot_reddening_vect(ax=ax[4, 0], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
helper_func.plot_reddening_vect(ax=ax[4, 1], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[4, 2], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[5, 0], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
helper_func.plot_reddening_vect(ax=ax[5, 1], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
helper_func.plot_reddening_vect(ax=ax[5, 2], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='b',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)



catalog_access = photometry_tools.data_access.CatalogAccess()
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
nuv_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F275W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=av_value)
max_color_ext_nuvb = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=nuv_wave, wave2=b_wave, av=av_value)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=av_value)
max_color_ext_bv = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=b_wave, wave2=v_wave, av=av_value)

slope_av_vector_nuvbvi = ((nuvb_int + max_color_ext_nuvb) - nuvb_int) / ((vi_int + max_color_ext_vi) - vi_int)
angle_av_vector_nuvbvi = np.arctan(slope_av_vector_nuvbvi) * 180/np.pi
slope_av_vector_ubvi = ((ub_int + max_color_ext_ub) - ub_int) / ((vi_int + max_color_ext_vi) - vi_int)
angle_av_vector_ubvi = np.arctan(slope_av_vector_ubvi) * 180/np.pi
slope_av_vector_bvvi = ((bv_int + max_color_ext_bv) - bv_int) / ((vi_int + max_color_ext_vi) - vi_int)
angle_av_vector_bvvi = np.arctan(slope_av_vector_bvvi) * 180/np.pi

print('slope_av_vector_nuvbvi ', slope_av_vector_nuvbvi)
print('angle_av_vector_nuvbvi ', angle_av_vector_nuvbvi)
print('slope_av_vector_ubvi ', slope_av_vector_ubvi)
print('angle_av_vector_ubvi ', angle_av_vector_ubvi)
print('slope_av_vector_bvvi ', slope_av_vector_bvvi)
print('angle_av_vector_bvvi ', angle_av_vector_bvvi)

ax[0, 0].text(vi_int-0.12, nuvb_int+0.05,
              r'slope = %.2f' % (slope_av_vector_nuvbvi),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=angle_av_vector_nuvbvi, fontsize=fontsize - 5)
ax[2, 0].text(vi_int-0.12, ub_int+0.05,
              r'slope = %.2f' % (slope_av_vector_ubvi),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=angle_av_vector_ubvi, fontsize=fontsize - 5)
ax[4, 0].text(vi_int-0.12, ub_int+0.05,
              r'slope = %.2f' % (slope_av_vector_bvvi),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=angle_av_vector_bvvi, fontsize=fontsize - 5)




ax[0, 0].set_title('Class 1 (Human)', fontsize=fontsize)
ax[0, 1].set_title('Class 2 (Human)', fontsize=fontsize)
ax[0, 2].set_title('Compact Associations (Human)', fontsize=fontsize)

ax[1, 0].set_title('Class 1 (ML)', fontsize=fontsize)
ax[1, 1].set_title('Class 2 (ML)', fontsize=fontsize)
ax[1, 2].set_title('Compact Associations (ML)', fontsize=fontsize)

#
# ax[0, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
#               'N = %i, (%i fitted)' %
#               (sum(mask_class_2_hum), sum(mask_class_2_hum*in_hull_young_hum*mask_good_colors_ubvi_hum)),
#               horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
# ax[0, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
#               'N = %i, (%i fitted)' %
#               (sum(mask_class_3_hum), sum(mask_class_3_hum*in_hull_young_hum*mask_good_colors_ubvi_hum)),
#               horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
#
# ax[1, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
#               'N = %i, (%i fitted)' %
#               (sum(mask_class_2_ml), sum(mask_class_2_ml*in_hull_young_ml*mask_good_colors_ubvi_ml)),
#               horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
# ax[1, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
#               'N = %i, (%i fitted)' %
#               (sum(mask_class_3_ml), sum(mask_class_3_ml*in_hull_young_ml*mask_good_colors_ubvi_ml)),
#               horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
#

#
# ax[0, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Class 2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
# ax[0, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Compact associations (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
# ax[1, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Class 1 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
# ax[1, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Class 2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
# ax[1, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Compact associations (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)


ax[0, 0].set_xlim(x_lim_vi)
ax[0, 1].set_xlim(x_lim_vi)
ax[0, 2].set_xlim(x_lim_vi)
ax[0, 0].set_ylim(y_lim_ub)
ax[0, 1].set_ylim(y_lim_ub)
ax[0, 2].set_ylim(y_lim_ub)

ax[1, 0].set_xlim(x_lim_vi)
ax[1, 1].set_xlim(x_lim_vi)
ax[1, 0].set_ylim(y_lim_ub)
ax[1, 1].set_ylim(y_lim_ub)

ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])
ax[0, 2].set_xticklabels([])

ax[0, 1].set_yticklabels([])
ax[0, 2].set_yticklabels([])
ax[1, 1].set_yticklabels([])
ax[1, 2].set_yticklabels([])
ax[2, 1].set_yticklabels([])
ax[2, 2].set_yticklabels([])
ax[3, 1].set_yticklabels([])
ax[3, 2].set_yticklabels([])
ax[4, 1].set_yticklabels([])
ax[4, 2].set_yticklabels([])
ax[5, 1].set_yticklabels([])
ax[5, 2].set_yticklabels([])

ax[0, 0].set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[2, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 0].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)
ax[5, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[5, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[5, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax[0, 0].legend(frameon=False, loc=3, bbox_to_anchor=(0, 0.05), fontsize=fontsize-3)

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
ax[4, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[4, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[4, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[5, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[5, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[5, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)




# plt.show()
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('plot_output/fit_reddening_vect_all.png')
plt.savefig('plot_output/fit_reddening_vect_all.pdf')


exit()


catalog_access = photometry_tools.data_access.CatalogAccess()
vi_int = 1.1
ub_int = -1.4
max_av = 1
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi_ccm89 = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub_ccm89 = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)
max_color_ext_vi_f99 = dust_tools.extinction_tools.ExtinctionTools.color_ext_f99_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub_f99 = dust_tools.extinction_tools.ExtinctionTools.color_ext_f99_av(wave1=u_wave, wave2=b_wave, av=max_av)

slope_av_vector_ccm89 = ((ub_int + max_color_ext_ub_ccm89) - ub_int) / ((vi_int + max_color_ext_vi_ccm89) - vi_int)
slope_av_vector_f99 = ((ub_int + max_color_ext_ub_f99) - ub_int) / ((vi_int + max_color_ext_vi_f99) - vi_int)



print('CCM (1989) Milky Way & %.2f \\\\ ' % slope_av_vector_ccm89)
print('F (1999) & %.2f \\\\ ' % slope_av_vector_f99)
print('Class 2 (Hum) & $%.2f\\pm%.2f$ \\\\ ' % (lin_fit_result_ubvi_c2_hum['gradient'], lin_fit_result_ubvi_c2_hum['gradient_err']))
print('Class 2 (ML) & $%.2f\\pm%.2f$ \\\\ ' % (lin_fit_result_ubvi_c2_ml['gradient'], lin_fit_result_ubvi_c2_ml['gradient_err']))
print('Compact associations (Hum) & $%.2f\\pm%.2f$ \\\\ ' % (lin_fit_result_ubvi_c3_hum['gradient'], lin_fit_result_ubvi_c3_hum['gradient_err']))
print('Compact associations (ML) & $%.2f\\pm%.2f$ \\\\ ' % (lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['gradient_err']))



