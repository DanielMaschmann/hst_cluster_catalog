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

# load models
age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
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
y_lim_bv = (1.7, -0.7)
n_bins_ubvi = 150
n_bins_bvvi = 150
kernel_size = 4.0


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

gauss_dict_ubvi_ml_1 = np.load('data_output/gauss_dict_ubvi_ml_1_thresh_10.npy', allow_pickle=True).item()
gauss_dict_bvvi_ml_1 = np.load('data_output/gauss_dict_bvvi_ml_1_thresh_10.npy', allow_pickle=True).item()

hull_gc_ubvi_ml_1_thresh_3 = fits.open('data_output/convex_hull_gc_ubvi_thres3_ml.fits')[1].data
hull_gc_ubvi_ml_1_thresh_10 = fits.open('data_output/convex_hull_gc_ubvi_thres10_ml.fits')[1].data
hull_gc_bvvi_ml_1_thresh_3 = fits.open('data_output/convex_hull_gc_bvvi_thres3_ml.fits')[1].data
hull_gc_bvvi_ml_1_thresh_10 = fits.open('data_output/convex_hull_gc_bvvi_thres10_ml.fits')[1].data

vi_hull_gc_ubvi_ml_1_thresh_3 = hull_gc_ubvi_ml_1_thresh_3['vi']
ub_hull_gc_ubvi_ml_1_thresh_3 = hull_gc_ubvi_ml_1_thresh_3['ub']
vi_hull_gc_ubvi_ml_1_thresh_10 = hull_gc_ubvi_ml_1_thresh_10['vi']
ub_hull_gc_ubvi_ml_1_thresh_10 = hull_gc_ubvi_ml_1_thresh_10['ub']

vi_hull_gc_bvvi_ml_1_thresh_3 = hull_gc_bvvi_ml_1_thresh_3['vi']
bv_hull_gc_bvvi_ml_1_thresh_3 = hull_gc_bvvi_ml_1_thresh_3['bv']
vi_hull_gc_bvvi_ml_1_thresh_10 = hull_gc_bvvi_ml_1_thresh_10['vi']
bv_hull_gc_bvvi_ml_1_thresh_10 = hull_gc_bvvi_ml_1_thresh_10['bv']


gauss_map_ubvi_ml_1 = gauss_dict_ubvi_ml_1['gauss_map']
seg_map_ubvi_ml_1 = gauss_dict_ubvi_ml_1['seg_deb_map']
bkg_map_ubvi_ml_1 = gauss_map_ubvi_ml_1.copy()
gc_map_ubvi_ml_1 = gauss_map_ubvi_ml_1.copy()
cascade_map_ubvi_ml_1 = gauss_map_ubvi_ml_1.copy()
bkg_map_ubvi_ml_1[seg_map_ubvi_ml_1._data != 0] = np.nan
gc_map_ubvi_ml_1[seg_map_ubvi_ml_1._data != 2] = np.nan
cascade_map_ubvi_ml_1[seg_map_ubvi_ml_1._data != 1] = np.nan

gauss_map_bvvi_ml_1 = gauss_dict_bvvi_ml_1['gauss_map']
seg_map_bvvi_ml_1 = gauss_dict_bvvi_ml_1['seg_deb_map']
bkg_map_bvvi_ml_1 = gauss_map_bvvi_ml_1.copy()
gc_map_bvvi_ml_1 = gauss_map_bvvi_ml_1.copy()
cascade_map_bvvi_ml_1 = gauss_map_bvvi_ml_1.copy()
bkg_map_bvvi_ml_1[seg_map_bvvi_ml_1._data != 0] = np.nan
gc_map_bvvi_ml_1[seg_map_bvvi_ml_1._data != 2] = np.nan
cascade_map_bvvi_ml_1[seg_map_bvvi_ml_1._data != 1] = np.nan

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(15, 20))
fontsize = 27

scale_ubvi_ml_1 = np.nanmax(gauss_dict_ubvi_ml_1['gauss_map'])
scale_bvvi_ml_1 = np.nanmax(gauss_dict_bvvi_ml_1['gauss_map'])

ax[0].imshow(bkg_map_ubvi_ml_1, extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
             origin='lower', interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=scale_ubvi_ml_1)
ax[0].imshow(gc_map_ubvi_ml_1, extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
             origin='lower', interpolation='nearest', aspect='auto', cmap='Reds', vmin=0, vmax=scale_ubvi_ml_1)
ax[0].imshow(cascade_map_ubvi_ml_1, extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
             origin='lower', interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=scale_ubvi_ml_1)

ax[1].imshow(bkg_map_bvvi_ml_1, extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
             origin='lower', interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=scale_bvvi_ml_1)
ax[1].imshow(gc_map_bvvi_ml_1, extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
             origin='lower', interpolation='nearest', aspect='auto', cmap='Reds', vmin=0, vmax=scale_bvvi_ml_1)
ax[1].imshow(cascade_map_bvvi_ml_1, extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
             origin='lower', interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=scale_bvvi_ml_1)

ax[0].plot(vi_hull_gc_ubvi_ml_1_thresh_10, ub_hull_gc_ubvi_ml_1_thresh_10, color='r', linewidth=4)
ax[0].plot(vi_hull_gc_ubvi_ml_1_thresh_3, ub_hull_gc_ubvi_ml_1_thresh_3, color='m', linewidth=4)

ax[1].plot(vi_hull_gc_bvvi_ml_1_thresh_10, bv_hull_gc_bvvi_ml_1_thresh_10, color='r', linewidth=4)
ax[1].plot(vi_hull_gc_bvvi_ml_1_thresh_3, bv_hull_gc_bvvi_ml_1_thresh_3, color='m', linewidth=4)

ax[0].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-', label=r'BC03, Z$_{\odot}$')
ax[0].plot(model_vi_sol50, model_ub_sol50, color='k', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}/50$')
ax[1].plot(model_vi_sol, model_bv_sol, color='darkred', linewidth=4, linestyle='-')
ax[1].plot(model_vi_sol50, model_bv_sol50, color='k', linewidth=4, linestyle='--')

ax[0].text(model_vi_sol[age_mod_sol == 1], model_ub_sol[age_mod_sol == 1]-0.15, r'1 Myr',
                       horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize)
ax[0].text(model_vi_sol[age_mod_sol == 5]+0.1, model_ub_sol[age_mod_sol == 5]+0.1, r'5 Myr',
                   horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax[0].text(model_vi_sol[age_mod_sol == 10]+0.03, model_ub_sol[age_mod_sol == 10]-0.07, r'10 Myr',
                   horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0].text(model_vi_sol[age_mod_sol == 100]-0.05, model_ub_sol[age_mod_sol == 100]+0.1, r'100 Myr',
                   horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax[0].text(model_vi_sol[age_mod_sol == 1000]-0.1, model_ub_sol[age_mod_sol == 1000]+0.1, r'1 Gyr',
                   horizontalalignment='right', verticalalignment='top', fontsize=fontsize)
ax[0].text(model_vi_sol[age_mod_sol == 10308] + 0.01, model_ub_sol[age_mod_sol == 10308]-0.1, r'10 Gyr',
                   horizontalalignment='left', verticalalignment='top', fontsize=fontsize)
ax[0].text(model_vi_sol[age_mod_sol == 13750]+0.01, model_ub_sol[age_mod_sol == 13750]+0.05, r'13 Gyr',
                   horizontalalignment='left', verticalalignment='top', fontsize=fontsize)


ax[1].text(model_vi_sol[age_mod_sol == 1], model_bv_sol[age_mod_sol == 1]-0.15, r'1 Myr',
                       horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize)
ax[1].text(model_vi_sol[age_mod_sol == 5]+0.1, model_bv_sol[age_mod_sol == 5]+0.1, r'5 Myr',
                   horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax[1].text(model_vi_sol[age_mod_sol == 10]+0.03, model_bv_sol[age_mod_sol == 10]-0.07, r'10 Myr',
                   horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1].text(model_vi_sol[age_mod_sol == 100]-0.05, model_bv_sol[age_mod_sol == 100]+0.1, r'100 Myr',
                   horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax[1].text(model_vi_sol[age_mod_sol == 1000]-0.03, model_bv_sol[age_mod_sol == 1000]+0.0, r'1 Gyr',
                   horizontalalignment='right', verticalalignment='top', fontsize=fontsize)
ax[1].text(model_vi_sol[age_mod_sol == 10308] + 0.01, model_bv_sol[age_mod_sol == 10308]-0.1, r'10 Gyr',
                   horizontalalignment='left', verticalalignment='top', fontsize=fontsize)
ax[1].text(model_vi_sol[age_mod_sol == 13750]+0.01, model_bv_sol[age_mod_sol == 13750]+0.05, r'13 Gyr',
                   horizontalalignment='left', verticalalignment='top', fontsize=fontsize)


ax[1].plot([1.335, 1.335], [1.5, 0.234], color='k', linewidth=2, linestyle='--')
ax[1].plot([1.335, 2.0], [0.234, 0.234], color='k', linewidth=2, linestyle='--')


vi_int = 1.2
ub_int = -1.5
bv_int = -0.2
av_value = 1
hf.plot_reddening_vect(ax=ax[0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize,
                       x_text_offset=0.05, y_text_offset=-0.05)

hf.plot_reddening_vect(ax=ax[1], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize,
                       x_text_offset=0.05, y_text_offset=-0.05)


x_lim_vi = (-0.6, 2.5)
y_lim_ub = (1.5, -2.2)
y_lim_bv = (1.7, -0.7)


ax[0].set_xlim(-0.6, 1.8)
ax[1].set_xlim(-0.6, 1.8)
ax[0].set_ylim(1.1, -2.2)
ax[1].set_ylim(1.3, -0.6)

ax[0].set_xticklabels([])

ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', labelpad=20, fontsize=fontsize)
ax[1].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


ax[0].set_title('Class 1 (ML)', fontsize=fontsize)

ax[0].legend(frameon=False, loc=3, fontsize=fontsize)

ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

# plt.show()
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('plot_output/represent_gc_region_ml.png')
fig.clf()
plt.cla()

exit()

