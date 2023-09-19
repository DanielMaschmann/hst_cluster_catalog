from photometry_tools import helper_func as hf
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import plotting_tools

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.spatial import ConvexHull
import multicolorfits as mcf

from photutils.segmentation import make_2dgaussian_kernel
from scipy.stats import gaussian_kde
from astropy.convolution import convolve
from matplotlib.patches import ConnectionPatch



age_mod_sol = np.load('../../color_color/data_output/age_mod_sol.npy')
model_ub_sol = np.load('../../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../../color_color/data_output/age_mod_sol50.npy')
model_ub_sol50 = np.load('../../color_color/data_output/model_ub_sol50.npy')
model_vi_sol50 = np.load('../../color_color/data_output/model_vi_sol50.npy')

ub_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.05, -0.1], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}

ub_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.1, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}


def display_models(ax, y_color='nuvb',
                   age_cut_sol50=5e2,
                   age_dots_sol=None,
                   age_dots_sol50=None,
                   age_labels=False,
                   age_label_color='k',
                   age_label_fontsize=30,
                   color_sol='darkred', linewidth_sol=4, linestyle_sol='-',
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
        ax.scatter(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age], color='k', s=80, zorder=20)

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




target_name_hum = np.load('../../color_color/data_output/target_name_hum.npy')
x_hum = np.load('../../color_color/data_output/x_hum.npy')
y_hum = np.load('../../color_color/data_output/y_hum.npy')
ra_hum = np.load('../../color_color/data_output/ra_hum.npy')
dec_hum = np.load('../../color_color/data_output/dec_hum.npy')
color_vi_hum = np.load('../../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../../color_color/data_output/color_ub_hum.npy')
color_vi_err_hum = np.load('../../color_color/data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('../../color_color/data_output/color_ub_err_hum.npy')
detect_u_hum = np.load('../../color_color/data_output/detect_u_hum.npy')
detect_b_hum = np.load('../../color_color/data_output/detect_b_hum.npy')
detect_v_hum = np.load('../../color_color/data_output/detect_v_hum.npy')
detect_i_hum = np.load('../../color_color/data_output/detect_i_hum.npy')
clcl_color_hum = np.load('../../color_color/data_output/clcl_color_hum.npy')
age_hum = np.load('../../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../../color_color/data_output/ebv_hum.npy')

target_name_ml = np.load('../../color_color/data_output/target_name_ml.npy')
x_ml = np.load('../../color_color/data_output/x_ml.npy')
y_ml = np.load('../../color_color/data_output/y_ml.npy')
ra_ml = np.load('../../color_color/data_output/ra_ml.npy')
dec_ml = np.load('../../color_color/data_output/dec_ml.npy')
color_vi_ml = np.load('../../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../../color_color/data_output/color_ub_ml.npy')
color_vi_err_ml = np.load('../../color_color/data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('../../color_color/data_output/color_ub_err_ml.npy')
detect_u_ml = np.load('../../color_color/data_output/detect_u_ml.npy')
detect_b_ml = np.load('../../color_color/data_output/detect_b_ml.npy')
detect_v_ml = np.load('../../color_color/data_output/detect_v_ml.npy')
detect_i_ml = np.load('../../color_color/data_output/detect_i_ml.npy')
clcl_color_ml = np.load('../../color_color/data_output/clcl_color_ml.npy')
clcl_qual_color_ml = np.load('../../color_color/data_output/clcl_qual_color_ml.npy')
age_ml = np.load('../../color_color/data_output/age_ml.npy')
ebv_ml = np.load('../../color_color/data_output/ebv_ml.npy')
mag_mask_ml = np.load('../../color_color/data_output/mag_mask_ml.npy')

# color range limitations
mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3
mask_class_12_hum = (clcl_color_hum == 1) | (clcl_color_hum == 2)

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3
mask_class_12_ml = (clcl_color_ml == 1) | (clcl_color_ml == 2)

mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml

x_lim_vi = (-0.7, 1.7)
y_lim_nuvb = (2.4, -2.9)
y_lim_ub = (1.3, -2.2)

mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                             (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                             mask_detect_ubvi_hum)
mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                            (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                            mask_detect_ubvi_ml)

target = 'ngc1097'
galaxy_name = 'ngc1097'

# get gauss und segmentations
n_bins_ubvi = 100
kernal_std = 3

red_color = '#FF2400'
green_color = '#0FFF50'
blue_color = '#0096FF'

red_color_hst = '#FF2400'
green_color_hst = '#0FFF50'
blue_color_hst = '#0096FF'
ha_color_hst = '#FF00FF'


n_bins = 1000
add_side_pixel_frac = -10/100
kernel_size_prop = 0.01

target_mask = (target_name_ml == target)
mask_old_ml = (age_ml > 10000) & target_mask
mask_mid_ml = ((age_ml > 30) & (age_ml < 100)) & target_mask
mask_young_ml = (age_ml < 5) & target_mask



n_added_pix = int(n_bins * add_side_pixel_frac)

# get min and max boundaries
left_ra = np.nanmax(ra_ml[target_mask])
right_ra = np.nanmin(ra_ml[target_mask])
lower_dec = np.nanmin(dec_ml[target_mask])
upper_dec = np.nanmax(dec_ml[target_mask])
min_x = np.nanmin(x_ml[target_mask])
max_x = np.nanmax(x_ml[target_mask])
min_y = np.nanmin(y_ml[target_mask])
max_y = np.nanmax(y_ml[target_mask])

x_bin_width = (max_x - min_x) / n_bins
y_bin_width = (max_y - min_y) / n_bins

center_ra = (left_ra + right_ra) / 2
center_dec = (lower_dec + upper_dec) / 2

pos_coord_lower_left = SkyCoord(ra=left_ra*u.deg, dec=lower_dec*u.deg)
pos_coord_lower_right = SkyCoord(ra=right_ra*u.deg, dec=lower_dec*u.deg)
pos_coord_upper_left = SkyCoord(ra=left_ra*u.deg, dec=upper_dec*u.deg)

ra_bin_width = (pos_coord_lower_left.separation(pos_coord_lower_right) / n_bins).degree
dec_bin_width = (pos_coord_lower_left.separation(pos_coord_upper_left) / n_bins).degree

x_bins = np.linspace(np.nanmin(x_ml[target_mask]) - x_bin_width*n_added_pix,
                     np.nanmax(x_ml[target_mask]) + x_bin_width*n_added_pix, n_bins + 1 + 2*n_added_pix)
y_bins = np.linspace(np.nanmin(y_ml[target_mask]) - y_bin_width*n_added_pix,
                     np.nanmax(y_ml[target_mask]) + y_bin_width*n_added_pix, n_bins + 1 + 2*n_added_pix)


# create histogram
hist_gc = np.histogram2d(x_ml[mask_old_ml], y_ml[mask_old_ml], bins=(x_bins, y_bins))[0].T
hist_cascade = np.histogram2d(x_ml[mask_mid_ml], y_ml[mask_mid_ml], bins=(x_bins, y_bins))[0].T
hist_young = np.histogram2d(x_ml[mask_young_ml], y_ml[mask_young_ml], bins=(x_bins, y_bins))[0].T

# now create a WCS for this histogram
wcs_hist = WCS(naxis=2)
# what is the center pixel of the XY grid.
wcs_hist.wcs.crpix = [hist_young.shape[0]/2, hist_young.shape[1]/2]
# what is the galactic coordinate of that pixel.
wcs_hist.wcs.crval = [center_ra, center_dec]
# what is the pixel scale in lon, lat.
wcs_hist.wcs.cdelt = np.array([-ra_bin_width, dec_bin_width])
# you would have to determine if this is in fact a tangential projection.
wcs_hist.wcs.ctype = ["RA---AIR", "DEC--AIR"]

# convolve with a square kernel
kernel_size = int(hist_young.shape[0] * kernel_size_prop) * 9
if (kernel_size % 2) == 0:
    kernel_size += 1

kernel = make_2dgaussian_kernel(hist_young.shape[0] * kernel_size_prop, size=kernel_size)

# hist_gc /= np.sum(hist_gc)
# hist_cascade /= np.sum(hist_cascade)
# hist_young /= np.sum(hist_young)

print(np.sum(hist_gc))
print(np.sum(hist_cascade))
print(np.sum(hist_young))

hist_gc = convolve(hist_gc, kernel)
hist_cascade = convolve(hist_cascade, kernel)
hist_young = convolve(hist_young, kernel)

grey_gc = mcf.greyRGBize_image(hist_gc, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                  gamma=4.0, checkscale=False)
grey_cascade = mcf.greyRGBize_image(hist_cascade, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                  gamma=4.0, checkscale=False)
grey_young = mcf.greyRGBize_image(hist_young, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                  gamma=4.0, checkscale=False)
r = mcf.colorize_image(grey_gc, red_color, colorintype='hex', gammacorr_color=2.8)
g = mcf.colorize_image(grey_cascade, green_color, colorintype='hex', gammacorr_color=2.8)
b = mcf.colorize_image(grey_young, blue_color, colorintype='hex', gammacorr_color=2.8)

rgb_image = mcf.combine_multicolor([r, g, b], gamma=2.8, inverse=False)
# make background white
mask_0_rgb_image = ((rgb_image[:, :, 0] < 15 / np.sum(hist_gc)) &
                    (rgb_image[:, :, 1] < 15 / np.sum(hist_cascade)) &
                    (rgb_image[:, :, 2] < 15 / np.sum(hist_young)))
rgb_image[mask_0_rgb_image] = 1.0

# gte alma observation data
alma_hdu = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' %
                     (galaxy_name, galaxy_name))
alma_wcs = WCS(alma_hdu[0].header)

alma_data_reprojected = plotting_tools.reproject_image(data=alma_hdu[0].data, wcs=alma_wcs,
                                                       new_wcs=wcs_hist, new_shape=rgb_image[:,:,0].shape)
alma_borders_reprojected = np.invert(np.isnan(alma_data_reprojected))



hst_band_list = ['F336W', 'F438W', 'F555W', 'F657N']
# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name=galaxy_name,
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')
# load all data
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='MJy/sr', band_list=hst_band_list, load_err=False)

central_coordinates = SkyCoord('2h46m20.0s -30d16m32s', unit=(u.hourangle, u.deg))
# get a cutout
cutout_dict_large_rgb = phangs_photometry.get_band_cutout_dict(ra_cutout=central_coordinates.ra.to(u.deg),
                                                               dec_cutout=central_coordinates.dec.to(u.deg),
                                                               cutout_size=(400, 400), include_err=False,
                                                               band_list=hst_band_list)
img_r_hst = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F555W_img_cutout'].data,
                                           wcs=cutout_dict_large_rgb['F555W_img_cutout'].wcs,
                                           new_wcs=wcs_hist, new_shape=rgb_image[:,:,0].shape)
img_g_hst = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F438W_img_cutout'].data,
                                           wcs=cutout_dict_large_rgb['F438W_img_cutout'].wcs,
                                           new_wcs=wcs_hist, new_shape=rgb_image[:,:,0].shape)
img_b_hst = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F336W_img_cutout'].data,
                                           wcs=cutout_dict_large_rgb['F336W_img_cutout'].wcs,
                                           new_wcs=wcs_hist, new_shape=rgb_image[:,:,0].shape)
rgb_image_hst = hf.create3color_rgb_img(img_r=img_r_hst, img_g=img_g_hst, img_b=img_b_hst,
                                        gamma=2.2, min_max=None, gamma_rgb=2.2,
                                        red_color=red_color_hst, green_color=green_color_hst, blue_color=blue_color_hst,
                                        mask_no_coverage=True)


# get zoom in
zoom_in_cutout_size = (40, 40)
cutout_dict_zoom_in = phangs_photometry.get_band_cutout_dict(ra_cutout=central_coordinates.ra.to(u.deg),
                                                               dec_cutout=central_coordinates.dec.to(u.deg),
                                                               cutout_size=zoom_in_cutout_size, include_err=False,
                                                               band_list=hst_band_list)

rgb_image_hst_zoom_in = hf.create3color_rgb_img(img_r=cutout_dict_zoom_in['F555W_img_cutout'].data,
                                                img_g=cutout_dict_zoom_in['F438W_img_cutout'].data,
                                                img_b=cutout_dict_zoom_in['F336W_img_cutout'].data,
                                                gamma=2.2, min_max=[0.01, 99.3], gamma_rgb=2.2,
                                                red_color=red_color_hst, green_color=green_color_hst,
                                                blue_color=blue_color_hst,
                                                mask_no_coverage=True)

coords_clusters_old = SkyCoord(ra=ra_ml[mask_old_ml]*u.deg, dec=dec_ml[mask_old_ml]*u.deg)
coords_clusters_mid = SkyCoord(ra=ra_ml[mask_mid_ml]*u.deg, dec=dec_ml[mask_mid_ml]*u.deg)
coords_clusters_young = SkyCoord(ra=ra_ml[mask_young_ml]*u.deg, dec=dec_ml[mask_young_ml]*u.deg)

pos_cluster_zoom_in_old = cutout_dict_zoom_in['F555W_img_cutout'].wcs.world_to_pixel(coords_clusters_old)
pos_cluster_zoom_in_mid = cutout_dict_zoom_in['F555W_img_cutout'].wcs.world_to_pixel(coords_clusters_mid)
pos_cluster_zoom_in_young = cutout_dict_zoom_in['F555W_img_cutout'].wcs.world_to_pixel(coords_clusters_young)

# get color color map
gauss_dict_ubvi_hum_12 = hf.gauss_weight_map(x_data=color_vi_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                             y_data=color_ub_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                             y_data_err=color_ub_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             kernal_std=kernal_std)



figure = plt.figure(figsize=(43, 15))
fontsize = 40
ax_rgb_hst = figure.add_axes([-0.02, 0.075, 0.40, 0.88], projection=wcs_hist)
ax_cluster = figure.add_axes([0.29, 0.075, 0.40, 0.88], projection=wcs_hist)
ax_rgb_hst_zoom_in = figure.add_axes([0.07, 0.02, 0.5, 0.5], projection=cutout_dict_zoom_in['F555W_img_cutout'].wcs)
ax_cc = figure.add_axes([0.695, 0.075, 0.30, 0.905])

ax_rgb_hst.imshow(rgb_image_hst)
ax_cluster.imshow(rgb_image)
print(rgb_image_hst.shape)



ax_rgb_hst_zoom_in.imshow(rgb_image_hst_zoom_in)
mask_in_zoom_img_old = ((pos_cluster_zoom_in_old[0] > 1) &
                        (pos_cluster_zoom_in_old[0] < (rgb_image_hst_zoom_in.shape[0]-1)) &
                        (pos_cluster_zoom_in_old[1] > 1) &
                        (pos_cluster_zoom_in_old[1] < (rgb_image_hst_zoom_in.shape[1]-1)))
mask_in_zoom_img_mid = ((pos_cluster_zoom_in_mid[0] > 1) &
                        (pos_cluster_zoom_in_mid[0] < (rgb_image_hst_zoom_in.shape[0]-1)) &
                        (pos_cluster_zoom_in_mid[1] > 1) &
                        (pos_cluster_zoom_in_mid[1] < (rgb_image_hst_zoom_in.shape[1]-1)))
mask_in_zoom_img_young = ((pos_cluster_zoom_in_young[0] > 0) &
                        (pos_cluster_zoom_in_young[0] < (rgb_image_hst_zoom_in.shape[0]-1)) &
                        (pos_cluster_zoom_in_young[1] > 0) &
                        (pos_cluster_zoom_in_young[1] < (rgb_image_hst_zoom_in.shape[1]-1)))

plotting_tools.plot_coord_circle(ax=ax_rgb_hst_zoom_in, pos=list(coords_clusters_old[mask_in_zoom_img_old]),
                                 rad=0.7, color=red_color, linewidth=1)
plotting_tools.plot_coord_circle(ax=ax_rgb_hst_zoom_in, pos=list(coords_clusters_mid[mask_in_zoom_img_mid]),
                                 rad=0.7, color=green_color, linewidth=1)
plotting_tools.plot_coord_circle(ax=ax_rgb_hst_zoom_in, pos=list(coords_clusters_young[mask_in_zoom_img_young]),
                                 rad=0.7, color=blue_color, linewidth=1)
ax_rgb_hst_zoom_in.axis('off')

plotting_tools.draw_box(ax=ax_rgb_hst, wcs=wcs_hist,
                        coord=central_coordinates, box_size=zoom_in_cutout_size,
                        color='white', linewidth=2, linestyle='--')
plotting_tools.draw_box(ax=ax_cluster, wcs=wcs_hist,
                        coord=central_coordinates, box_size=zoom_in_cutout_size,
                        color='k', linewidth=3, linestyle='--')

pos_edge_1 = SkyCoord(ra=central_coordinates.ra.to(u.deg) + (zoom_in_cutout_size[0]*u.arcsec / 2)/np.cos(central_coordinates.dec.degree*np.pi/180),
                      dec=central_coordinates.dec.to(u.deg) - zoom_in_cutout_size[1]*u.arcsec / 2)
pos_pix_edge_1 = wcs_hist.world_to_pixel(pos_edge_1)
con_spec_edge_1 = ConnectionPatch(
    xyA=(pos_pix_edge_1[0], pos_pix_edge_1[1]), coordsA=ax_rgb_hst.transData,
    xyB=(ax_rgb_hst_zoom_in.get_xlim()[0], ax_rgb_hst_zoom_in.get_ylim()[0]), coordsB=ax_rgb_hst_zoom_in.transData,
    linestyle="-", linewidth=3, color='grey')
figure.add_artist(con_spec_edge_1)

pos_edge_2 = SkyCoord(ra=central_coordinates.ra.to(u.deg) - (zoom_in_cutout_size[0]*u.arcsec / 2)/np.cos(central_coordinates.dec.degree*np.pi/180),
                      dec=central_coordinates.dec.to(u.deg) + zoom_in_cutout_size[1]*u.arcsec / 2)
pos_pix_edge_2 = wcs_hist.world_to_pixel(pos_edge_2)
con_spec_edge_2 = ConnectionPatch(
    xyA=(pos_pix_edge_2[0], pos_pix_edge_2[1]), coordsA=ax_rgb_hst.transData,
    xyB=(ax_rgb_hst_zoom_in.get_xlim()[1], ax_rgb_hst_zoom_in.get_ylim()[1]), coordsB=ax_rgb_hst_zoom_in.transData,
    linestyle="-", linewidth=3, color='grey')
figure.add_artist(con_spec_edge_2)



ax_rgb_hst.text(rgb_image_hst[:,:,0].shape[0]*0.05,
                rgb_image_hst[:,:,0].shape[1]*0.95,
                'U F336W', color=blue_color, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_rgb_hst.text(rgb_image_hst[:,:,0].shape[0]*0.05,
                rgb_image_hst[:,:,0].shape[1]*0.91,
                'B F438W', color=green_color, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_rgb_hst.text(rgb_image_hst[:,:,0].shape[0]*0.05,
                rgb_image_hst[:,:,0].shape[1]*0.87,
                'V F555W', color=red_color, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax_cluster.scatter([],[], color=blue_color, s=150, label=r'age < 5 Myr')
ax_cluster.scatter([],[], color=green_color, s=150, label=r'30 Myr < age < 100 Myr')
ax_cluster.scatter([],[], color=red_color, s=150, label=r'age > 10 Gyr')

ax_cluster.legend(frameon=True, facecolor='darkgrey', edgecolor='k', loc=4, fontsize=fontsize-3)

ax_rgb_hst.set_title('NGC 1097  HST rgb Image', fontsize=fontsize)
ax_cluster.set_title('NGC 1097  Spatial Cluster Distribution', fontsize=fontsize)


scale = np.nanmax(gauss_dict_ubvi_hum_12['gauss_map'])
ax_cc.imshow(gauss_dict_ubvi_hum_12['gauss_map'], origin='lower',
                    extent=(np.min(gauss_dict_ubvi_hum_12['x_bins_gauss']), np.max(gauss_dict_ubvi_hum_12['x_bins_gauss']),
                            np.min(gauss_dict_ubvi_hum_12['y_bins_gauss']), np.max(gauss_dict_ubvi_hum_12['y_bins_gauss'])),
                    cmap='Greys', vmin=0, vmax=scale, interpolation='nearest', aspect='auto')

ax_cc.scatter(color_vi_ml[mask_young_ml], color_ub_ml[mask_young_ml], color=blue_color)
ax_cc.scatter(color_vi_ml[mask_mid_ml], color_ub_ml[mask_mid_ml], color=green_color)
ax_cc.scatter(color_vi_ml[mask_old_ml], color_ub_ml[mask_old_ml], color=red_color)



vi_int = 1.0
ub_int = -1.8
av_value = 1
hf.plot_reddening_vect(ax=ax_cc, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       x_text_offset=0.05, y_text_offset=-0.1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize)
display_models(ax=ax_cc, age_label_fontsize=fontsize+2, y_color='ub', age_labels=True,
               label_sol=r'BC03, Z$_{\odot}$',
               label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')

ax_cc.legend(frameon=False, loc=3, bbox_to_anchor=(0, 0.0), fontsize=fontsize-3)

ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc.set_ylabel('U (F336W) - B (F438W)', fontsize=fontsize)

ax_cc.set_xlim(x_lim_vi)
ax_cc.set_ylim(y_lim_ub)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plotting_tools.arr_axis_params(ax=ax_rgb_hst, ra_tick_label=True, dec_tick_label=True,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                    ra_minpad=0.4, dec_minpad=0.4, tick_color='k', label_color='k',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)

plotting_tools.arr_axis_params(ax=ax_cluster, ra_tick_label=True, dec_tick_label=False,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                    ra_minpad=0.4, dec_minpad=0.4, tick_color='k', label_color='k',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)


# plt.show()
plt.savefig('plot_output/cluster_ngc1098.png')
plt.savefig('plot_output/cluster_ngc1098.pdf')




exit()
# create RGB image
rgb_image = make_lupton_rgb(cutout_dict_large_rgb['F657N_img_cutout'].data, cutout_dict_large_rgb['F555W_img_cutout'].data,
                            cutout_dict_large_rgb['F438W_img_cutout'].data, Q=14, stretch=3)


# now identify the objects in this region
all_coordinates = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
separation = central_coordinates.separation(all_coordinates)

close_mask = separation < 10*u.arcsec
# get young objects
selection_young = np.where(((clcl == 2) | (clcl == 3))
                           & (clcl_qual >= 0.8)
                           & (age < 10)
                           # & (ebv > 0.3)
                           & (mstar > 1e1) & (chi2 < 1.2) & close_mask
                           )
selection_inter = np.where(((clcl == 2) | (clcl == 3))
                           & (clcl_qual >= 0.8)
                           & (age >= 10) & (age < 100)
                           & (ebv > 0.3) & (mstar > 1e1) & (chi2 < 1.2) & close_mask
                           )
selection_old = np.where((clcl == 1) & (clcl_qual >= 0.8) & (age > 1000) & (ebv < 0.1) & (mstar > 1e3) & (chi2 < 1.2) & close_mask)

#indexes = list(selection_young[0]) + list(selection_inter[0]) + list(selection_old[0])
# index_old = 940 # dusty old globular cluster
# index_old = 1329 # also dusty old cluster
# index_old = 997
index_old = 1099
# index_inter = selection_inter[0][0]
index_inter = 1116 # very young dsty cluster
index_yellow = 807
# index_young = selection_young[0][0]
index_young = 1094
print('index_old ', index_old)
print('index_inter ', index_inter)
print('index_young ', index_young)

result_dict_old = fit_cigale(index=index_old, met=0.0004, ebv=[0.0])
result_dict_inter = fit_cigale(index=index_inter, met=0.02)
result_dict_yellow = fit_cigale(index=index_yellow, met=0.02)
result_dict_young = fit_cigale(index=index_young, met=0.02)
print(result_dict_old)
print(result_dict_inter)
print(result_dict_yellow)
print(result_dict_young)


pos_old = SkyCoord(ra=ra[index_old] * u.deg, dec=dec[index_old] * u.deg)
pos_inter = SkyCoord(ra=ra[index_inter] * u.deg, dec=dec[index_inter] * u.deg)
pos_yellow = SkyCoord(ra=ra[index_yellow] * u.deg, dec=dec[index_yellow] * u.deg)
pos_young = SkyCoord(ra=ra[index_young] * u.deg, dec=dec[index_young] * u.deg)

cutout_dict_old = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_old.ra.to(u.deg),
                                                         dec_cutout=pos_old.dec.to(u.deg),
                                                         cutout_size=(4, 4), include_err=False, band_list=hst_band_list)
rgb_old = make_lupton_rgb(cutout_dict_old['F657N_img_cutout'].data, cutout_dict_old['F555W_img_cutout'].data,
                          cutout_dict_old['F438W_img_cutout'].data, Q=14, stretch=3)
cutout_dict_inter = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_inter.ra.to(u.deg),
                                                           dec_cutout=pos_inter.dec.to(u.deg),
                                                           cutout_size=(4, 4), include_err=False,
                                                           band_list=hst_band_list)
rgb_inter = make_lupton_rgb(cutout_dict_inter['F657N_img_cutout'].data, cutout_dict_inter['F555W_img_cutout'].data,
                            cutout_dict_inter['F438W_img_cutout'].data, Q=14, stretch=3)
cutout_dict_yellow = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_yellow.ra.to(u.deg),
                                                            dec_cutout=pos_yellow.dec.to(u.deg),
                                                            cutout_size=(4, 4), include_err=False,
                                                            band_list=hst_band_list)
rgb_yellow = make_lupton_rgb(cutout_dict_yellow['F657N_img_cutout'].data, cutout_dict_yellow['F555W_img_cutout'].data,
                             cutout_dict_yellow['F438W_img_cutout'].data, Q=14, stretch=3)
cutout_dict_young = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_young.ra.to(u.deg),
                                                           dec_cutout=pos_young.dec.to(u.deg),
                                                           cutout_size=(4, 4), include_err=False,
                                                           band_list=hst_band_list)
rgb_young = make_lupton_rgb(cutout_dict_young['F657N_img_cutout'].data, cutout_dict_young['F555W_img_cutout'].data,
                            cutout_dict_young['F438W_img_cutout'].data, Q=14, stretch=3)


figure = plt.figure(figsize=(35, 11))
fontsize = 26
ax_rgb = figure.add_axes([0.02, 0.08, 0.3, 0.905], projection=cutout_dict_large_rgb['F555W_img_cutout'].wcs)
ax_rgb_old = figure.add_axes([-0.03, 0.4, 0.2, 0.2], projection=cutout_dict_old['F555W_img_cutout'].wcs)
ax_rgb_yellow = figure.add_axes([0.19, 0.1, 0.2, 0.2], projection=cutout_dict_inter['F555W_img_cutout'].wcs)
ax_rgb_inter = figure.add_axes([0.05, 0.71, 0.2, 0.2], projection=cutout_dict_inter['F555W_img_cutout'].wcs)
ax_rgb_young = figure.add_axes([0.20, 0.74, 0.2, 0.2], projection=cutout_dict_young['F555W_img_cutout'].wcs)

ax_sed = figure.add_axes([0.36, 0.075, 0.37, 0.905])
ax_cc_vi_ub = figure.add_axes([0.78, 0.54, 0.21, 0.45])
ax_cc_vi_nuvb = figure.add_axes([0.78, 0.075, 0.21, 0.45])

circ_rad_arcsec = 1

ax_rgb.imshow(rgb_image)

# add names for colors
ax_rgb.text(100, 150, r'R: F657N (H$\alpha$)', color='lightcoral', fontsize=fontsize)
ax_rgb.text(100, 100, 'G: F555W', color='springgreen', fontsize=fontsize)
ax_rgb.text(100, 50, 'B: F438W', color='royalblue', fontsize=fontsize)
ax_rgb.text(60, rgb_image.shape[0]-100, 'NGC 1097', color='white', fontsize=fontsize+5)

circle_old = SphericalCircle(pos_old, circ_rad_arcsec * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                             alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_old)
circle_yellow = SphericalCircle(pos_yellow, circ_rad_arcsec * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                                alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_yellow)
circle_inter = SphericalCircle(pos_inter, circ_rad_arcsec * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_inter)
circle_young = SphericalCircle(pos_young, circ_rad_arcsec * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_young)

ax_rgb_old.imshow(rgb_old)
ax_rgb_yellow.imshow(rgb_yellow)
ax_rgb_inter.imshow(rgb_inter)
ax_rgb_young.imshow(rgb_young)

ax_rgb_old.set_title('C1', fontsize=fontsize+5, color='r')
ax_rgb_yellow.set_title('C3', fontsize=fontsize+5, color='y')
ax_rgb_inter.set_title('C2', fontsize=fontsize+5, color='green')
ax_rgb_young.set_title('C4', fontsize=fontsize+5, color='blue')

plotting_tools.arr_axis_params(ax=ax_rgb, ra_tick_label=True, dec_tick_label=True,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_rgb_old, ra_tick_label=False, dec_tick_label=False,
                    ra_axis_label=' ', dec_axis_label=' ',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_rgb_yellow, ra_tick_label=False, dec_tick_label=False,
                    ra_axis_label=' ', dec_axis_label=' ',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_rgb_inter, ra_tick_label=False, dec_tick_label=False,
                    ra_axis_label=' ', dec_axis_label=' ',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_rgb_young, ra_tick_label=False, dec_tick_label=False,
                    ra_axis_label=' ', dec_axis_label=' ',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)

circ_rad_x = hf.transform_world2pix_scale(length_in_arcsec=circ_rad_arcsec,
                                          wcs=cutout_dict_large_rgb['F555W_img_cutout'].wcs)
circ_rad_y = hf.transform_world2pix_scale(length_in_arcsec=circ_rad_arcsec,
                                          wcs=cutout_dict_large_rgb['F555W_img_cutout'].wcs, dim=1)
print('circ_rad_x ', circ_rad_x)
print('circ_rad_y ', circ_rad_y)

pos_pix_circ_old = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_old)
con_spec_old_1 = ConnectionPatch(
    xyA=(pos_pix_circ_old[0] - circ_rad_x, pos_pix_circ_old[1]), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_old.get_xlim()[0], ax_rgb_old.get_ylim()[1]), coordsB=ax_rgb_old.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_old_1)
con_spec_old_2 = ConnectionPatch(
    xyA=(pos_pix_circ_old[0] + circ_rad_x, pos_pix_circ_old[1]), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_old.get_xlim()[1], ax_rgb_old.get_ylim()[1]), coordsB=ax_rgb_old.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_old_2)


pos_pix_circ_yellow = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_yellow)
con_spec_yellow_1 = ConnectionPatch(
    xyA=(pos_pix_circ_yellow[0], pos_pix_circ_yellow[1] - circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_yellow.get_xlim()[0], ax_rgb_yellow.get_ylim()[0]), coordsB=ax_rgb_yellow.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_yellow_1)
con_spec_yellow_2 = ConnectionPatch(
    xyA=(pos_pix_circ_yellow[0], pos_pix_circ_yellow[1] + circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_yellow.get_xlim()[0], ax_rgb_yellow.get_ylim()[1]), coordsB=ax_rgb_yellow.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_yellow_2)


pos_pix_circ_inter = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_inter)
con_spec_inter_1 = ConnectionPatch(
    xyA=(pos_pix_circ_inter[0], pos_pix_circ_inter[1] - circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_inter.get_xlim()[1], ax_rgb_inter.get_ylim()[0]), coordsB=ax_rgb_inter.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_inter_1)
con_spec_inter_2 = ConnectionPatch(
    xyA=(pos_pix_circ_inter[0], pos_pix_circ_inter[1] + circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_inter.get_xlim()[1], ax_rgb_inter.get_ylim()[1]), coordsB=ax_rgb_inter.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_inter_2)


pos_pix_circ_young = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_young)
con_spec_young_1 = ConnectionPatch(
    xyA=(pos_pix_circ_young[0] + circ_rad_x, pos_pix_circ_young[1]), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_young.get_xlim()[0], ax_rgb_young.get_ylim()[0]), coordsB=ax_rgb_young.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_young_1)
con_spec_young_2 = ConnectionPatch(
    xyA=(pos_pix_circ_young[0], pos_pix_circ_young[1] + circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_young.get_xlim()[0], ax_rgb_young.get_ylim()[1]), coordsB=ax_rgb_young.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_young_2)



ax_sed.plot(result_dict_old['wavelength'] * 1e-3, result_dict_old['stellar_spectrum'], linewidth=3, color='k')
ax_sed.plot(result_dict_yellow['wavelength'] * 1e-3, result_dict_yellow['stellar_spectrum'], linewidth=3, color='k')
ax_sed.plot(result_dict_inter['wavelength'] * 1e-3, result_dict_inter['stellar_spectrum'], linewidth=3, color='k')
ax_sed.plot(result_dict_young['wavelength'] * 1e-3, result_dict_young['stellar_spectrum'], linewidth=3, color='k')

plot_data_points(ax=ax_sed, index=index_old, color='r', size=13,
                 label=r'C1, age=%i Gyr, M$_{*}$ = %i 10$^{5}$ M$_{\odot}$, Z=Z$_{\odot}$/50, E(B-V)=%.2f' %
                       (result_dict_old['best_age'][0]/1000, round(result_dict_old['best_m_star'][0]/1e5),
                        result_dict_old['best_ebv'][0]))
plot_data_points(ax=ax_sed, index=index_yellow, color='y', size=13,
                 label=r'C3, age=%i Myr, M$_{*}$ = %i 10$^{5}$ M$_{\odot}$, Z=Z$_{\odot}$, E(B-V)=%.2f' %
                       (result_dict_yellow['best_age'][0], round(result_dict_yellow['best_m_star'][0]/1e5),
                        result_dict_yellow['best_ebv'][0]))
plot_data_points(ax=ax_sed, index=index_inter, color='g', size=13,
                  label=r'C2, age=%i Myr, M$_{*}$ = %i 10$^{5}$ M$_{\odot}$, Z=Z$_{\odot}$, E(B-V)=%.2f' %
                       (result_dict_inter['best_age'][0], round(result_dict_inter['best_m_star'][0]/1e5),
                        result_dict_inter['best_ebv'][0]))
plot_data_points(ax=ax_sed, index=index_young, color='b', size=13,
                 label=r'C4, age=%i Myr, M$_{*}$ = %i 10$^{5}$ M$_{\odot}$, Z=Z$_{\odot}$, E(B-V)=%.2f' %
                       (result_dict_young['best_age'][0], round(result_dict_young['best_m_star'][0]/1e5),
                        result_dict_young['best_ebv'][0]))




ax_sed.legend(frameon=False, fontsize=fontsize)
ax_sed.set_ylabel(r'S$_{\nu}$ (mJy)', fontsize=fontsize)
ax_sed.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)
ax_sed.set_xlim(200 * 1e-3, 1.5)
ax_sed.set_ylim(0.000009, 3e2)
ax_sed.set_xscale('log')
ax_sed.set_yscale('log')
ax_sed.set_xlabel(r'Observed ${\lambda}$ ($\mu$m)', labelpad=-8, fontsize=fontsize)


ax_cc_vi_ub.plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='-', alpha=1.0, label='BC03, Z=Z$_{\odot}$')
ax_cc_vi_ub.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='-', alpha=1.0, label='BC03, Z=Z$_{\odot}$/50')
ax_cc_vi_ub.scatter(color_vi_ml_12[index_old], color_ub_ml_12[index_old], color='r', s=150, zorder=10)
ax_cc_vi_ub.scatter(color_vi_ml_12[index_inter], color_ub_ml_12[index_inter], color='g', s=150, zorder=10)
ax_cc_vi_ub.scatter(color_vi_ml_12[index_yellow], color_ub_ml_12[index_yellow], color='y', s=150, zorder=10)
ax_cc_vi_ub.scatter(color_vi_ml_12[index_young], color_ub_ml_12[index_young], color='b', s=150, zorder=10)

ax_cc_vi_nuvb.plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=3, linestyle='-', alpha=1.0)
ax_cc_vi_nuvb.plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=3, linestyle='-', alpha=1.0)
ax_cc_vi_nuvb.scatter(color_vi_ml_12[index_old], color_nuvb_ml_12[index_old], color='r', s=150, zorder=10)
ax_cc_vi_nuvb.scatter(color_vi_ml_12[index_inter], color_nuvb_ml_12[index_inter], color='g', s=150, zorder=10)
ax_cc_vi_nuvb.scatter(color_vi_ml_12[index_yellow], color_nuvb_ml_12[index_yellow], color='y', s=150, zorder=10)
ax_cc_vi_nuvb.scatter(color_vi_ml_12[index_young], color_nuvb_ml_12[index_young], color='b', s=150, zorder=10)

ax_cc_vi_ub.set_ylim(0.9, -1.9)
ax_cc_vi_ub.set_xlim(-0.4, 1.6)
ax_cc_vi_nuvb.set_ylim(2.6, -3.1)
ax_cc_vi_nuvb.set_xlim(-0.4, 1.6)

#ax_cc_vi_ub.set_xticks(ax_cc_vi_nuvb.get_xticks())
ax_cc_vi_ub.set_xticklabels([])
ax_cc_vi_nuvb.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_cc_vi_ub.set_ylabel('U (F336W) - B (F438W)', fontsize=fontsize)
ax_cc_vi_nuvb.set_ylabel('NUV (F275W) - B (F438W)', labelpad=30, fontsize=fontsize)

ax_cc_vi_ub.legend(frameon=False, fontsize=fontsize, loc=3)

ax_cc_vi_ub.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_vi_nuvb.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



plt.savefig('plot_output/sed_explain.png')
plt.savefig('plot_output/sed_explain.pdf')

exit()