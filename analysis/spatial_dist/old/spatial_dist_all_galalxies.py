import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf, plotting_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from scipy.stats import gaussian_kde

import dust_tools.extinction_tools
from astropy.convolution import convolve

from photutils.segmentation import make_2dgaussian_kernel
import multicolorfits as mcf

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from scipy.spatial import ConvexHull


target_name_hum = np.load('../color_color/data_output/target_name_hum.npy')
target_name_ml = np.load('../color_color/data_output/target_name_ml.npy')

color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
age_ml = np.load('../color_color/data_output/age_ml.npy')
ra_ml = np.load('../color_color/data_output/ra_ml.npy')
dec_ml = np.load('../color_color/data_output/dec_ml.npy')
x_ml = np.load('../color_color/data_output/x_ml.npy')
y_ml = np.load('../color_color/data_output/y_ml.npy')

clcl_color_ml = np.load('../color_color/data_output/clcl_color_ml.npy')


# get groups of the hull

convex_hull_gc_hum = fits.open('../color_color_regions/data_output/convex_hull_ubvi_12_gc_hum.fits')[1].data
convex_hull_cascade_hum = fits.open('../color_color_regions/data_output/convex_hull_ubvi_12_cascade_hum.fits')[1].data
convex_hull_young_hum = fits.open('../color_color_regions/data_output/convex_hull_ubvi_12_young_hum.fits')[1].data

convex_hull_gc_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_12_gc_ml.fits')[1].data
convex_hull_cascade_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_12_cascade_ml.fits')[1].data
convex_hull_young_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_12_young_ml.fits')[1].data

convex_hull_vi_gc_hum = convex_hull_gc_hum['vi']
convex_hull_ub_gc_hum = convex_hull_gc_hum['ub']
convex_hull_vi_cascade_hum = convex_hull_cascade_hum['vi']
convex_hull_ub_cascade_hum = convex_hull_cascade_hum['ub']
convex_hull_vi_young_hum = convex_hull_young_hum['vi']
convex_hull_ub_young_hum = convex_hull_young_hum['ub']

convex_hull_vi_gc_ml = convex_hull_gc_ml['vi']
convex_hull_ub_gc_ml = convex_hull_gc_ml['ub']
convex_hull_vi_cascade_ml = convex_hull_cascade_ml['vi']
convex_hull_ub_cascade_ml = convex_hull_cascade_ml['ub']
convex_hull_vi_young_ml = convex_hull_young_ml['vi']
convex_hull_ub_young_ml = convex_hull_young_ml['ub']

hull_gc_ml = ConvexHull(np.array([convex_hull_vi_gc_ml, convex_hull_ub_gc_ml]).T)
hull_cascade_ml = ConvexHull(np.array([convex_hull_vi_cascade_ml, convex_hull_ub_cascade_ml]).T)
hull_young_ml = ConvexHull(np.array([convex_hull_vi_young_ml, convex_hull_ub_young_ml]).T)


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
    dist_list.append(catalog_access.dist_dict[galaxy_name]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

n_bins = 100
kernal_std = 20
kernel = make_2dgaussian_kernel(kernal_std, size=51)  # FWHM = 3.0

for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    mask_target_ml = target_name_ml == target

    x_data_ml = x_ml[mask_target_ml]
    y_data_ml = y_ml[mask_target_ml]

    color_vi_data_ml = color_vi_ml[mask_target_ml]
    color_ub_data_ml = color_ub_ml[mask_target_ml]

    age_data_ml = age_ml[mask_target_ml]

    in_hull_gc_ml = hf.points_in_hull(np.array([color_vi_data_ml, color_ub_data_ml]).T, hull_gc_ml)
    in_hull_cascade_ml = hf.points_in_hull(np.array([color_vi_data_ml, color_ub_data_ml]).T, hull_cascade_ml)
    in_hull_young_ml = hf.points_in_hull(np.array([color_vi_data_ml, color_ub_data_ml]).T, hull_young_ml)

    young_cluster = age_data_ml < 10

    x_bins_hist = np.linspace(np.min(x_data_ml), np.max(x_data_ml), n_bins)
    y_bins_hist = np.linspace(np.min(y_data_ml), np.max(y_data_ml), n_bins)


    hist_gc_ml = np.histogram2d(x_data_ml[in_hull_gc_ml*np.invert(young_cluster)], y_data_ml[in_hull_gc_ml*np.invert(young_cluster)], bins=(x_bins_hist, y_bins_hist))[0]
    hist_cascade_ml = np.histogram2d(x_data_ml[in_hull_cascade_ml*np.invert(young_cluster)], y_data_ml[in_hull_cascade_ml*np.invert(young_cluster)], bins=(x_bins_hist, y_bins_hist))[0]
    hist_young_ml = np.histogram2d(x_data_ml[in_hull_young_ml + young_cluster], y_data_ml[in_hull_young_ml + young_cluster], bins=(x_bins_hist, y_bins_hist))[0]

    hist_gc_ml = hist_gc_ml / np.sum(hist_gc_ml)
    hist_cascade_ml = hist_cascade_ml / np.sum(hist_cascade_ml)
    hist_young_ml = hist_young_ml / np.sum(hist_young_ml)

    # conv_hist_gc_ml = convolve(hist_gc_ml, kernel)
    # conv_hist_cascade_ml = convolve(hist_cascade_ml, kernel)
    # conv_hist_young_ml = convolve(hist_young_ml, kernel)
    #

    grey_hst_r_ml = mcf.greyRGBize_image(hist_gc_ml, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                      gamma=4.0, checkscale=False)
    grey_hst_g_ml = mcf.greyRGBize_image(hist_cascade_ml, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                      gamma=4.0, checkscale=False)
    grey_hst_b_ml = mcf.greyRGBize_image(hist_young_ml, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                      gamma=4.0, checkscale=False)
    hst_r_purple_ml = mcf.colorize_image(grey_hst_r_ml, '#FF4433', colorintype='hex', gammacorr_color=2.8)
    hst_g_orange_ml = mcf.colorize_image(grey_hst_g_ml, '#0FFF50', colorintype='hex', gammacorr_color=2.8)
    hst_b_blue_ml = mcf.colorize_image(grey_hst_b_ml, '#0096FF', colorintype='hex', gammacorr_color=2.8)

    gc_image_ml = mcf.combine_multicolor([hst_r_purple_ml], gamma=2.8, inverse=False)
    cascade_image_ml = mcf.combine_multicolor([hst_g_orange_ml], gamma=2.8, inverse=False)
    young_image_ml = mcf.combine_multicolor([hst_b_blue_ml], gamma=2.8, inverse=False)

    rgb_hst_image_ml = mcf.combine_multicolor([hst_r_purple_ml, hst_g_orange_ml, hst_b_blue_ml], gamma=2.8, inverse=False)

    mask_0_gc_image_ml = (gc_image_ml[:,:, 0] == 0) & (gc_image_ml[:,:, 1] == 0) & (gc_image_ml[:,:, 2] == 0)
    mask_0_cascade_image_ml = (cascade_image_ml[:,:, 0] == 0) & (cascade_image_ml[:,:, 1] == 0) & (cascade_image_ml[:,:, 2] == 0)
    mask_0_young_image_ml = (young_image_ml[:,:, 0] == 0) & (young_image_ml[:,:, 1] == 0) & (young_image_ml[:,:, 2] == 0)
    mask_0_rgb_image_ml = (rgb_hst_image_ml[:,:, 0] == 0) & (rgb_hst_image_ml[:,:, 1] == 0) & (rgb_hst_image_ml[:,:, 2] == 0)


    gc_image_ml[mask_0_gc_image_ml] = 1.0
    cascade_image_ml[mask_0_cascade_image_ml] = 1.0
    young_image_ml[mask_0_young_image_ml] = 1.0
    rgb_hst_image_ml[mask_0_rgb_image_ml] = 1.0


    figure = plt.figure(figsize=(38, 10))
    fontsize = 20
    ax_img_gc = figure.add_axes([-0.30, 0.07, 0.88, 0.88]) # projection=cutout_alma.wcs)
    ax_img_cascade = figure.add_axes([-0.05, 0.07, 0.88, 0.88]) # projection=cutout_alma.wcs)
    ax_img_young = figure.add_axes([0.19, 0.07, 0.88, 0.88]) # projection=cutout_alma.wcs)
    ax_img_rgb = figure.add_axes([0.43, 0.07, 0.88, 0.88]) # projection=cutout_alma.wcs)


    ax_img_gc.imshow(gc_image_ml, origin='lower')
    ax_img_cascade.imshow(cascade_image_ml, origin='lower')
    ax_img_young.imshow(young_image_ml, origin='lower')

    ax_img_rgb.imshow(rgb_hst_image_ml, origin='lower')

    ax_img_gc.set_title('ML GC', fontsize=fontsize)
    ax_img_cascade.set_title('ML Cascade', fontsize=fontsize)
    ax_img_young.set_title('ML young', fontsize=fontsize)
    #
    # plotting_tools.arr_axis_params(ax=ax_img_gc, ra_tick_label=True, dec_tick_label=True,
    #                                ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
    #                                ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
    #                                fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    # plotting_tools.arr_axis_params(ax=ax_img_cascade, ra_tick_label=True, dec_tick_label=False,
    #                                ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
    #                                ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
    #                                fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    # plotting_tools.arr_axis_params(ax=ax_img_young, ra_tick_label=True, dec_tick_label=False,
    #                                ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
    #                                ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
    #                                fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    # plotting_tools.arr_axis_params(ax=ax_img_rgb, ra_tick_label=True, dec_tick_label=False,
    #                                ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
    #                                ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
    #                                fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    #

    plt.savefig('plot_output/composit_spatial_dist_%s.png' % target)


exit()






