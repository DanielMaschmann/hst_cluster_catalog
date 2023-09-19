import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from matplotlib.patches import Ellipse
import matplotlib
from matplotlib.colorbar import ColorbarBase
from matplotlib.collections import LineCollection
import dust_tools.extinction_tools
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from lmfit import Model, Parameters
from photutils.segmentation import deblend_sources
import math

import sep

from photutils.centroids import (centroid_1dg, centroid_2dg,

                                 centroid_com, centroid_quadratic)
from scipy.spatial import ConvexHull


def get_scale(map, x, y, a, b, theta, n_frac, max_counts):
    scale = 1
    gc_content = len(color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull])
    initial_step_size = 0.5
    count = 1
    print(gc_content)

    while True:

        mask_in_ell = hf.check_point_inside_ellipse(x_ell=x, y_ell=y, a_ell=a*scale, b_ell=b*scale, theta_ell=theta,
                                                    x_p=color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull],
                                                    y_p=color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull]) != 0
        ellipse_content = sum(mask_in_ell)
        frac = ellipse_content / gc_content

        if frac > n_frac:
            scale -= initial_step_size / count
        else:
            scale += initial_step_size / count

        is_accuracy = abs(frac - n_frac)
        # print(scale, is_accuracy, frac)
        count += 1
        if count > max_counts:
            break

    ax.scatter(color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull][mask_in_ell], color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull][mask_in_ell], c='r')
    ax.scatter(color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull][~mask_in_ell], color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull][~mask_in_ell], c='b')

    return frac, scale


# load models
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')

model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')

# load color data
color_vi_hum = np.load('../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../color_color/data_output/color_ub_hum.npy')
color_bv_hum = np.load('../color_color/data_output/color_bv_hum.npy')
color_vi_err_hum = np.load('../color_color/data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('../color_color/data_output/color_ub_err_hum.npy')
color_bv_err_hum = np.load('../color_color/data_output/color_bv_err_hum.npy')
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
color_vi_err_ml = np.load('../color_color/data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('../color_color/data_output/color_ub_err_ml.npy')
color_bv_err_ml = np.load('../color_color/data_output/color_bv_err_ml.npy')
detect_u_ml = np.load('../color_color/data_output/detect_u_ml.npy')
detect_b_ml = np.load('../color_color/data_output/detect_b_ml.npy')
detect_v_ml = np.load('../color_color/data_output/detect_v_ml.npy')
detect_i_ml = np.load('../color_color/data_output/detect_i_ml.npy')

clcl_color_ml = np.load('../color_color/data_output/clcl_color_ml.npy')
clcl_qual_color_ml = np.load('../color_color/data_output/clcl_qual_color_ml.npy')
age_ml = np.load('../color_color/data_output/age_ml.npy')
ebv_ml = np.load('../color_color/data_output/ebv_ml.npy')


mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_bvvi_hum = detect_b_hum * detect_v_hum * detect_i_hum

mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_detect_bvvi_ml = detect_b_ml * detect_v_ml * detect_i_ml


# color range limitations
x_lim_vi = (-0.6, 2.5)

y_lim_ub = (1.5, -2.2)
n_bins_ubvi = 150

y_lim_bv = (1.7, -0.7)
n_bins_bvvi = 150


mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_hum)
mask_good_colors_bvvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_bv_hum > (y_lim_bv[1] - 1)) & (color_bv_hum < (y_lim_bv[0] + 1)) &
                               mask_detect_bvvi_hum)

mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_ml)
mask_good_colors_bvvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_bv_ml > (y_lim_bv[1] - 1)) & (color_bv_ml < (y_lim_bv[0] + 1)) &
                               mask_detect_bvvi_ml)

# masks for classes
mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3
mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

threshold_fact = 10
gauss_dict_ubvi_ml_1 = hf.calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                    y_data=color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                     x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                     y_data_err=color_ub_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                     x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi, threshold_fact=threshold_fact)
convex_hull = fits.open('data_output/convex_hull_ubvi_thres10_ml.fits')[1].data




print(convex_hull.names)
convex_hull_vi = convex_hull['vi']
convex_hull_ub = convex_hull['ub']

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
fontsize = 17

gauss_map = gauss_dict_ubvi_ml_1['gauss_map']
seg_map = gauss_dict_ubvi_ml_1['seg_deb_map']

gc_map = gauss_map.copy()

gc_map[seg_map._data != 2] = np.nan
sep_table = sep.extract(data=gc_map, thresh=np.nanmax(gauss_dict_ubvi_ml_1['gauss_map'])/100)

# from photutils.segmentation import SourceCatalog
# cat = SourceCatalog(data=gauss_map, segment_img=seg_map)
# x = cat[1].to_table()['xcentroid'].value / gauss_map.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0]
# y = cat[1].to_table()['ycentroid'].value / gauss_map.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1]
# a = cat[1].to_table()['semimajor_sigma'].value / gauss_map.shape[0] * (x_lim_vi[1] - x_lim_vi[0])
# b = cat[1].to_table()['semiminor_sigma'].value / gauss_map.shape[1] * (y_lim_ub[1] - y_lim_ub[0])
# e = Ellipse(xy=(x, y), width = 1.1*a, height = 1.1*b, angle = cat[1].to_table()['orientation'].value*108/np.pi)
# e.set_edgecolor('red')
# e.set_facecolor('none')
# ax.add_artist(e)




# ax.imshow(gauss_map, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
#                         cmap='Greys', vmin=0, vmax=np.nanmax(gc_map)/1)

ax.imshow(gc_map / np.nanmax(gc_map), origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                        cmap='Reds', vmin=0, vmax=np.nanmax(gc_map)/1)

ax.plot(convex_hull_vi[:-1], convex_hull_ub[:-1])

x1, y1 = centroid_com(gc_map)

x1 = x1 / gauss_map.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0]
y1 = y1 / gauss_map.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1]
ax.scatter(x1, y1, color='tab:blue')


# get data points inside the hull

hull = ConvexHull(np.array([convex_hull_vi, convex_hull_ub]).T)
in_hull = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull)
# ax.scatter(color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull], color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml * in_hull])


x_bins = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_ubvi)
y_bins = np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins_ubvi)
x_grid, y_grid = np.meshgrid(x_bins, y_bins)

print(x_grid)

for i in range(len(sep_table)):




    x = sep_table['x'][i] / gauss_map.shape[0] * (x_lim_vi[1] - x_lim_vi[0]) + x_lim_vi[0]
    y = sep_table['y'][i] / gauss_map.shape[1] * (y_lim_ub[0] - y_lim_ub[1]) + y_lim_ub[1]
    a = sep_table['a'][i] / gauss_map.shape[0] * (x_lim_vi[1] - x_lim_vi[0])
    b = sep_table['b'][i] / gauss_map.shape[1] * (y_lim_ub[1] - y_lim_ub[0])
    theta = sep_table['theta'][i]

    frac_68, scale_68 = get_scale(map=gc_map, x=x, y=y, a=a, b=b, theta=theta, n_frac=0.68, max_counts=1000)
    frac_95, scale_95 = get_scale(map=gc_map, x=x, y=y, a=a, b=b, theta=theta, n_frac=0.95, max_counts=10000)
    print('frac_68 ', frac_68)
    print('frac_95 ', frac_95)

    e = Ellipse(xy=(x, y), width = scale_68*a, height = scale_68*b, angle = theta*180/np.pi)
    e.set_edgecolor('green')
    e.set_facecolor('none')
    ax.add_artist(e)

    e = Ellipse(xy=(x, y), width = scale_95*a, height = scale_95*b, angle = theta*180/np.pi)
    e.set_edgecolor('green')
    e.set_facecolor('none')
    ax.add_artist(e)

ax.set_xlim(x_lim_vi)
ax.set_ylim(y_lim_ub)

plt.show()

exit()


for threshold_fact in [2, 3, 4, 5, 6, 7, 8, 9, 10]:

    gauss_dict_ubvi_hum_1 = hf.calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                     y_data=color_ub_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                     x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                     y_data_err=color_ub_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                     x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi, threshold_fact=threshold_fact)
    gauss_dict_ubvi_ml_1 = hf.calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                    y_data=color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                     x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                     y_data_err=color_ub_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                     x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi, threshold_fact=threshold_fact)
    gauss_dict_bvvi_hum_1 = hf.calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                     y_data=color_bv_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                     x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                     y_data_err=color_bv_err_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                     x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi, threshold_fact=threshold_fact)
    gauss_dict_bvvi_ml_1 = hf.calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                    y_data=color_bv_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                    x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                    y_data_err=color_bv_err_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                    x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi, threshold_fact=threshold_fact)


    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
    fontsize = 17

    hf.plot_reg_map(ax=ax[0, 0], gauss_map=gauss_dict_ubvi_hum_1['gauss_map'], seg_map=gauss_dict_ubvi_hum_1['seg_deb_map'],
                 x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi, plot_cont_2=True, x_label='vi', y_label='ub', save_str_2='ubvi_thres%i_hum' % threshold_fact)
    hf.plot_reg_map(ax=ax[0, 1], gauss_map=gauss_dict_ubvi_ml_1['gauss_map'], seg_map=gauss_dict_ubvi_ml_1['seg_deb_map'],
                 x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi, plot_cont_2=True, x_label='vi', y_label='ub', save_str_2='ubvi_thres%i_ml' % threshold_fact)

    hf.plot_reg_map(ax=ax[1, 0], gauss_map=gauss_dict_bvvi_hum_1['gauss_map'], seg_map=gauss_dict_bvvi_hum_1['seg_deb_map'],
                 x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi, plot_cont_2=True, x_label='vi', y_label='bv', save_str_2='bvvi_thres%i_hum' % threshold_fact)
    hf.plot_reg_map(ax=ax[1, 1], gauss_map=gauss_dict_bvvi_ml_1['gauss_map'], seg_map=gauss_dict_bvvi_ml_1['seg_deb_map'],
                 x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi, plot_cont_2=True, x_label='vi', y_label='bv', save_str_2='bvvi_thres%i_ml' % threshold_fact)

    ax[0, 0].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[0, 0].plot(model_vi_sol50, model_ub_sol50, color='k', linewidth=2, linestyle='--', label=r'BC03, Z$_{\odot}/50$')

    ax[0, 1].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-')
    ax[0, 1].plot(model_vi_sol50, model_ub_sol50, color='k', linewidth=2, linestyle='--')

    ax[1, 0].plot(model_vi_sol, model_bv_sol, color='darkred', linewidth=2, linestyle='-')
    ax[1, 0].plot(model_vi_sol50, model_bv_sol50, color='k', linewidth=2, linestyle='--')

    ax[1, 1].plot(model_vi_sol, model_bv_sol, color='darkred', linewidth=2, linestyle='-')
    ax[1, 1].plot(model_vi_sol50, model_bv_sol50, color='k', linewidth=2, linestyle='--')


    ax[0, 0].set_title('Hum Class 1', fontsize=fontsize)
    ax[0, 1].set_title('ML Class 1', fontsize=fontsize)


    ax[0, 0].set_xlim(x_lim_vi)
    ax[0, 1].set_xlim(x_lim_vi)
    ax[0, 0].set_ylim(y_lim_ub)
    ax[0, 1].set_ylim(y_lim_ub)

    ax[1, 0].set_xlim(x_lim_vi)
    ax[1, 1].set_xlim(x_lim_vi)
    ax[1, 0].set_ylim(y_lim_bv)
    ax[1, 1].set_ylim(y_lim_bv)

    ax[0, 0].set_xticklabels([])
    ax[0, 1].set_xticklabels([])

    ax[0, 1].set_yticklabels([])
    ax[1, 1].set_yticklabels([])

    ax[0, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
    ax[1, 0].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)
    ax[1, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
    ax[1, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


    ax[0, 0].legend(frameon=False, loc=3, fontsize=fontsize)

    ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


    # plt.show()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('plot_output/regions_overview_seg_%i.png' % threshold_fact)
    fig.clf()
    plt.cla()

