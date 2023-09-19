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


#
# cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
# hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
# morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
# sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
#
# catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
#                                                             hst_obs_hdr_file_path=hst_obs_hdr_file_path,
#                                                             morph_mask_path=morph_mask_path,
#                                                             sample_table_path=sample_table_path)

target = 'ngc0628'
galaxy_name = 'ngc0628'
phangs_photometry = photometry_tools.analysis_tools.AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name=galaxy_name,
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')


simbad_table = Simbad.query_object(galaxy_name)
central_coordinates = SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
                               unit=(u.hourangle, u.deg))

cutout_size = (250, 250)


alma_hdu = fits.open('../overview_img/data/ngc0628_12m+7m+tp_co21_broad_tpeak.fits')
cutout_alma = hf.get_img_cutout(img=alma_hdu[0].data, wcs=WCS(alma_hdu[0].header),
                                         coord=central_coordinates, cutout_size=cutout_size)

# print(cutout_alma.data.shape)
#
# exit()
#
# band_list = ['F555W']
#
# phangs_photometry.load_hst_nircam_miri_bands(flux_unit='MJy/sr', band_list=band_list, load_err=False)
#
#
# # get a cutout
# cutout_dict_large_rgb = phangs_photometry.get_band_cutout_dict(ra_cutout=central_coordinates.ra.to(u.deg),
#                                                                dec_cutout=central_coordinates.dec.to(u.deg),
#                                                                cutout_size=cutout_size, include_err=False,
#                                                                band_list=band_list)

# hst_wcs = cutout_dict_large_rgb['F555W_img_cutout'].wcs
# hst_shape = cutout_dict_large_rgb['F555W_img_cutout'].data.shape




# hst_img = plotting_tools.reproject_image(data=cutout_alma.data, wcs=cutout_alma.wcs,
#                                           new_wcs=hst_wcs,
#                                           new_shape=new_shape)



#
#
# target_list = catalog_access.target_hst_cc
# dist_list = []
# for target in target_list:
#     if (target == 'ngc0628c') | (target == 'ngc0628e'):
#         target = 'ngc0628'
#     dist_list.append(catalog_access.dist_dict[target]['dist'])
# sort = np.argsort(dist_list)
# target_list = np.array(target_list)[sort]
# dist_list = np.array(dist_list)[sort]
#
# catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')
#
# color_c1 = 'darkorange'
# color_c2 = 'tab:green'
# color_c3 = 'darkorange'


# target = 'ngc0628c'



# cc_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
#
# ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
# ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')

# x_ml_12, y_ml_12 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml')
# x_ml_3, y_ml_3 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml')


color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
ra_ml = np.load('../color_color/data_output/ra_ml.npy')
dec_ml = np.load('../color_color/data_output/dec_ml.npy')
clcl_color_ml = np.load('../color_color/data_output/clcl_color_ml.npy')

# gte position
pos_ml = cutout_alma.wcs.world_to_pixel(SkyCoord(ra=ra_ml*u.deg, dec=dec_ml*u.deg))
x_ml, y_ml = pos_ml[0], pos_ml[1]

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

in_hull_gc_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_gc_ml)
in_hull_cascade_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_cascade_ml)
in_hull_young_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_young_ml)

#
# plt.scatter(color_vi_ml[(clcl_color_ml==1)*in_hull_gc_ml_1], color_ub_ml[(clcl_color_ml==1)*in_hull_gc_ml_1])
# plt.scatter(color_vi_ml[(clcl_color_ml==2)*in_hull_cascade_ml_2], color_ub_ml[(clcl_color_ml==2)*in_hull_cascade_ml_2])
# plt.scatter(color_vi_ml[(clcl_color_ml==3)*in_hull_young_ml_3], color_ub_ml[(clcl_color_ml==3)*in_hull_young_ml_3])
#
# plt.plot(convex_hull_vi_gc, convex_hull_ub_gc)
# plt.plot(convex_hull_vi_cascade, convex_hull_ub_cascade)
# plt.plot(convex_hull_vi_young, convex_hull_ub_young)
#
# plt.show()
#
#
#
#
# exit()

n_bins = 1000
kernal_std = 20

new_shape = cutout_alma.data.shape

x_bins_hist = np.linspace(0, new_shape[0], new_shape[0])
y_bins_hist = np.linspace(0, new_shape[1], new_shape[1])

hist_gc = np.histogram2d(x_ml[in_hull_gc_ml], y_ml[in_hull_gc_ml], bins=(x_bins_hist, y_bins_hist))[0]
hist_cascade = np.histogram2d(x_ml[in_hull_cascade_ml], y_ml[in_hull_cascade_ml], bins=(x_bins_hist, y_bins_hist))[0]
hist_young = np.histogram2d(x_ml[in_hull_young_ml], y_ml[in_hull_young_ml], bins=(x_bins_hist, y_bins_hist))[0]

hist_gc = hist_gc / np.sum(hist_gc)
hist_cascade = hist_cascade / np.sum(hist_cascade)
hist_young = hist_young / np.sum(hist_young)

kernel = make_2dgaussian_kernel(kernal_std, size=51)  # FWHM = 3.0
conv_hist_gc = convolve(hist_gc, kernel)
conv_hist_cascade = convolve(hist_cascade, kernel)
conv_hist_young = convolve(hist_young, kernel)


grey_hst_r = mcf.greyRGBize_image(conv_hist_gc, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                  gamma=4.0, checkscale=False)
grey_hst_g = mcf.greyRGBize_image(conv_hist_cascade, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                  gamma=4.0, checkscale=False)
grey_hst_b = mcf.greyRGBize_image(conv_hist_young, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                  gamma=4.0, checkscale=False)
hst_r_purple = mcf.colorize_image(grey_hst_r, '#FF4433', colorintype='hex', gammacorr_color=2.8)
hst_g_orange = mcf.colorize_image(grey_hst_g, '#0FFF50', colorintype='hex', gammacorr_color=2.8)
hst_b_blue = mcf.colorize_image(grey_hst_b, '#0096FF', colorintype='hex', gammacorr_color=2.8)

gc_image = mcf.combine_multicolor([hst_r_purple], gamma=2.8, inverse=False)
cascade_image = mcf.combine_multicolor([hst_g_orange], gamma=2.8, inverse=False)
young_image = mcf.combine_multicolor([hst_b_blue], gamma=2.8, inverse=False)

rgb_hst_image = mcf.combine_multicolor([hst_r_purple, hst_g_orange, hst_b_blue], gamma=2.8, inverse=False)


# fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 6))
# fontsize = 18

figure = plt.figure(figsize=(38, 10))
fontsize = 20
ax_img_gc = figure.add_axes([-0.30, 0.07, 0.88, 0.88], projection=cutout_alma.wcs)
ax_img_cascade = figure.add_axes([-0.05, 0.07, 0.88, 0.88], projection=cutout_alma.wcs)
ax_img_young = figure.add_axes([0.19, 0.07, 0.88, 0.88], projection=cutout_alma.wcs)
ax_img_rgb = figure.add_axes([0.43, 0.07, 0.88, 0.88], projection=cutout_alma.wcs)

#
# gc_image[gc_image < 0.001] = 1.0
# cascade_image[cascade_image < 0.001] = 1.0
# young_image[young_image < 0.001] = 1.0
# rgb_hst_image[rgb_hst_image < 0.001] = 1.0



ax_img_gc.imshow(gc_image)
ax_img_cascade.imshow(cascade_image)
ax_img_young.imshow(young_image)

ax_img_rgb.imshow(rgb_hst_image)
ax_img_rgb.contour(cutout_alma.data, levels=[0.8, 1, 1.3, 1.6, 1.9], colors='white', alpha=0.5, linewidth=2)


ax_img_gc.set_title('ML GC', fontsize=fontsize)
ax_img_cascade.set_title('ML Cascade', fontsize=fontsize)
ax_img_young.set_title('ML young', fontsize=fontsize)

plotting_tools.arr_axis_params(ax=ax_img_gc, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_cascade, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_young, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_rgb, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)


plt.savefig('plot_output/composit_spatial_dist.png')


exit()






