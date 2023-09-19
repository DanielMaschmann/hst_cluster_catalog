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
from photometry_tools import helper_func


def sort_counterclockwise(points, centre = None):
  if centre:
    centre_x, centre_y = centre
  else:
    centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)
  angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
  counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
  counterclockwise_points = [points[i] for i in counterclockwise_indices]
  return counterclockwise_points


def gauss2d(x, y, x0, y0, sig_x, sig_y):
    expo = -(((x - x0)**2)/(2 * sig_x**2) + ((y - y0)**2)/(2 * sig_y**2))
    norm_amp = 1 / (2 * np.pi * sig_x * sig_y)
    return norm_amp * np.exp(expo)



def calc_seg(x_data, y_data, x_data_err, y_data_err, x_lim, y_lim, n_bins, threshold_fact=2, kernel_fwhm=3.0):

    # calculate combined errors
    data_err = np.sqrt(x_data_err**2 + y_data_err**2)
    noise_cut = np.percentile(data_err, 90)

    # bins
    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh
    x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
    gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
    noise_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))

    for color_index in range(len(x_data)):
        gauss = gauss2d(x=x_mesh, y=y_mesh, x0=x_data[color_index], y0=y_data[color_index],
                        sig_x=x_data_err[color_index], sig_y=y_data_err[color_index])
        gauss_map += gauss
        if data_err[color_index] > noise_cut:
            noise_map += gauss

    # subtract median noise
    gauss_map -= np.nanmean(noise_map)

    # gaussian convolution
    kernel = make_2dgaussian_kernel(kernel_fwhm, size=5)  # FWHM = 3.0
    conv_gauss_map = convolve(gauss_map, kernel)

    # get a segment detection threshold
    threshold = len(x_data) / threshold_fact
    seg_map = detect_sources(conv_gauss_map, threshold, npixels=50)
    seg_deb_map = deblend_sources(conv_gauss_map, seg_map, npixels=50, nlevels=32, contrast=0.1, progress_bar=False)

    return_dict = {
        'gauss_map': gauss_map, 'conv_gauss_map': conv_gauss_map, 'seg_deb_map': seg_deb_map}

    return return_dict



def plot_reg_map(ax, gauss_map, seg_map, x_lim, y_lim, n_bins, save_string, contour_index=0):

    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh

    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    conv_gauss_map = convolve(gauss_map, kernel)
    gauss_map_no_seg = conv_gauss_map.copy()
    gauss_map_seg1 = conv_gauss_map.copy()
    gauss_map_seg2 = conv_gauss_map.copy()
    # gauss_map_seg3 = conv_gauss_map.copy()
    gauss_map_no_seg[seg_map._data != 0] = np.nan
    gauss_map_seg1[seg_map._data != 1] = np.nan
    gauss_map_seg2[seg_map._data != 2] = np.nan
    # gauss_map_seg3[seg_map._data != 3] = np.nan
    ax.imshow(gauss_map_no_seg, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                    cmap='Greys', vmin=0, vmax=np.nanmax(conv_gauss_map)/1)
    ax.imshow(gauss_map_seg1, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                    cmap='Greens', vmin=0, vmax=np.nanmax(gauss_map_seg1)/1)
    ax.imshow(gauss_map_seg2, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                    cmap='Blues', vmin=0, vmax=np.nanmax(gauss_map_seg2)/1)
    # ax.imshow(gauss_map_seg3, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
    #                 cmap='Reds', vmin=0, vmax=np.nanmax(gauss_map_seg3)/1)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    cs = ax.contour(x_bins_gauss, y_bins_gauss, (seg_map._data == 2),
                    colors='k', linewidth=2, levels=[0.1])
    p = cs.collections[0].get_paths()[contour_index]
    v = p.vertices
    # get all points from contour
    x_cont = []
    y_cont = []
    for point in v:
        x_cont.append(point[0])
        y_cont.append(point[1])

    x_cont = np.array(x_cont)
    y_cont = np.array(y_cont)
    counterclockwise_points = sort_counterclockwise(points=np.array([x_cont, y_cont]).T)

    counterclockwise_points = np.array(counterclockwise_points)

    x_convex_hull = counterclockwise_points[:, 0]
    y_convex_hull = counterclockwise_points[:, 1]

    np.save('data_output/x_convex_hull_%s.npy' % save_string, x_convex_hull)
    np.save('data_output/y_convex_hull_%s.npy' % save_string, y_convex_hull)

    ax.scatter(x_convex_hull, y_convex_hull)



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






gauss_dict_ubvi_hum_1 = calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                 y_data=color_ub_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                 x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                 y_data_err=color_ub_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                 x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi, threshold_fact=8)
gauss_dict_ubvi_ml_1 = calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                y_data=color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                 x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                 y_data_err=color_ub_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                 x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi, threshold_fact=8)
gauss_dict_bvvi_hum_1 = calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                 y_data=color_bv_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                 x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                 y_data_err=color_bv_err_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                                 x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi, threshold_fact=8)
gauss_dict_bvvi_ml_1 = calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                y_data=color_bv_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                y_data_err=color_bv_err_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                                x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi, threshold_fact=8)




fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
fontsize = 17


plot_reg_map(ax=ax[0, 0], gauss_map=gauss_dict_ubvi_hum_1['gauss_map'], seg_map=gauss_dict_ubvi_hum_1['seg_deb_map'],
             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi, save_string='_ubvi_hum', contour_index=0)
plot_reg_map(ax=ax[0, 1], gauss_map=gauss_dict_ubvi_ml_1['gauss_map'], seg_map=gauss_dict_ubvi_ml_1['seg_deb_map'],
             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi, save_string='_ubvi_ml', contour_index=0)

plot_reg_map(ax=ax[1, 0], gauss_map=gauss_dict_bvvi_hum_1['gauss_map'], seg_map=gauss_dict_bvvi_hum_1['seg_deb_map'],
             x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi, save_string='_bvvi_hum', contour_index=0)
plot_reg_map(ax=ax[1, 1], gauss_map=gauss_dict_bvvi_ml_1['gauss_map'], seg_map=gauss_dict_bvvi_ml_1['seg_deb_map'],
             x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins_bvvi, save_string='_bvvi_ml', contour_index=0)

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
plt.savefig('plot_output/regions_overview_seg.png')

