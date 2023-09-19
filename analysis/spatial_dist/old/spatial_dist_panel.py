import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf, plotting_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from scipy.stats import gaussian_kde
from astropy.stats import sigma_clipped_stats

import dust_tools.extinction_tools
from astropy.convolution import convolve

from photutils.segmentation import make_2dgaussian_kernel
import multicolorfits as mcf

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from scipy.spatial import ConvexHull



def get_rgb_images(kernal_size_prop, shape, x, y, mask_r, mask_g, mask_b):
    x_bins_hist = np.linspace(0, shape[0], shape[0])
    y_bins_hist = np.linspace(0, shape[1], shape[1])

    hist_r = np.histogram2d(x[mask_r], y[mask_r], bins=(x_bins_hist, y_bins_hist))[0]
    hist_g = np.histogram2d(x[mask_g], y[mask_g], bins=(x_bins_hist, y_bins_hist))[0]
    hist_b = np.histogram2d(x[mask_b], y[mask_b], bins=(x_bins_hist, y_bins_hist))[0]

    # convolve with a square kernel
    kernal_size = int(shape[0] * kernal_size_prop)
    if (kernal_size%2) == 0:
        kernal_size += 1
    square_kernel = np.ones((kernal_size, kernal_size))
    hist_r = convolve(hist_r, square_kernel)
    hist_g = convolve(hist_g, square_kernel)
    hist_b = convolve(hist_b, square_kernel)

    hist_r /= np.sum(hist_r)
    hist_g /= np.sum(hist_g)
    hist_b /= np.sum(hist_b)

    grey_r = mcf.greyRGBize_image(hist_r, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                      gamma=4.0, checkscale=False)
    grey_g = mcf.greyRGBize_image(hist_g, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                      gamma=4.0, checkscale=False)
    grey_b = mcf.greyRGBize_image(hist_b, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                      gamma=4.0, checkscale=False)
    r = mcf.colorize_image(grey_r, '#FF4433', colorintype='hex', gammacorr_color=2.8)
    g = mcf.colorize_image(grey_g, '#0FFF50', colorintype='hex', gammacorr_color=2.8)
    b = mcf.colorize_image(grey_b, '#0096FF', colorintype='hex', gammacorr_color=2.8)

    r_image = mcf.combine_multicolor([r], gamma=2.8, inverse=False)
    g_image = mcf.combine_multicolor([g], gamma=2.8, inverse=False)
    b_image = mcf.combine_multicolor([b], gamma=2.8, inverse=False)
    rgb_image = mcf.combine_multicolor([r, g, b], gamma=2.8, inverse=False)


    # make background white
    mask_0_r_image = (r_image[:,:, 0] == 0) & (r_image[:,:, 1] == 0) & (r_image[:,:, 2] == 0)
    mask_0_g_image = (g_image[:,:, 0] == 0) & (g_image[:,:, 1] == 0) & (g_image[:,:, 2] == 0)
    mask_0_b_image = (b_image[:,:, 0] == 0) & (b_image[:,:, 1] == 0) & (b_image[:,:, 2] == 0)
    mask_0_rgb_image = (rgb_image[:,:, 0] == 0) & (rgb_image[:,:, 1] == 0) & (rgb_image[:,:, 2] == 0)

    r_image[mask_0_r_image] = 1.0
    g_image[mask_0_g_image] = 1.0
    b_image[mask_0_b_image] = 1.0
    rgb_image[mask_0_rgb_image] = 1.0

    return r_image, g_image, b_image, rgb_image


def plot_panels(ax_img_r, ax_img_g, ax_img_b, ax_img_rgb, img_r, img_g, img_b, img_rgb,
                contour_data, contour_color='gold', alpha=1.0, linewidths=2):
    ax_img_r.imshow(img_r)
    ax_img_g.imshow(img_g)
    ax_img_b.imshow(img_b)
    ax_img_rgb.imshow(img_rgb)
    mean, median, std = sigma_clipped_stats(contour_data, sigma=3.0)
    levels_std = np.array(np.arange(start=0.3, stop=5, step=1), dtype=float)
    levels = np.ones(len(levels_std)) * mean
    levels += levels_std
    ax_img_b.contour(contour_data,
                               levels=levels,
                               colors=contour_color, alpha=alpha, linewidths=linewidths)


# get alma data
simbad_table_ngc0628 = Simbad.query_object('ngc0628')
central_coordinates_ngc0628 = SkyCoord('%s %s' %
                                       (simbad_table_ngc0628['RA'].value[0],  simbad_table_ngc0628['DEC'].value[0]),
                                       unit=(u.hourangle, u.deg))
simbad_table_ngc1097 = Simbad.query_object('ngc1097')
central_coordinates_ngc1097 = SkyCoord('%s %s' %
                                       (simbad_table_ngc1097['RA'].value[0],  simbad_table_ngc1097['DEC'].value[0]),
                                       unit=(u.hourangle, u.deg))
simbad_table_ngc1566 = Simbad.query_object('ngc1566')
central_coordinates_ngc1566 = SkyCoord('%s %s' %
                                       (simbad_table_ngc1566['RA'].value[0],  simbad_table_ngc1566['DEC'].value[0]),
                                       unit=(u.hourangle, u.deg))

cutout_size_ngc0628 = (250, 250)
cutout_size_ngc1097 = (250, 250)
cutout_size_ngc1566 = (172.5, 172.5)
alma_hdu_ngc0628 = fits.open('/home/benutzer/data/PHANGS-ALMA/ngc0628/ngc0628_12m+7m+tp_co21_broad_tpeak.fits')
alma_hdu_ngc1097 = fits.open('/home/benutzer/data/PHANGS-ALMA/ngc1097/ngc1097_12m+7m+tp_co21_broad_tpeak.fits')
alma_hdu_ngc1566 = fits.open('/home/benutzer/data/PHANGS-ALMA/ngc1566/ngc1566_12m+7m+tp_co21_broad_tpeak.fits')
cutout_alma_ngc0628 = hf.get_img_cutout(img=alma_hdu_ngc0628[0].data, wcs=WCS(alma_hdu_ngc0628[0].header),
                                        coord=central_coordinates_ngc0628, cutout_size=cutout_size_ngc0628)
cutout_alma_ngc1097 = hf.get_img_cutout(img=alma_hdu_ngc1097[0].data, wcs=WCS(alma_hdu_ngc1097[0].header),
                                        coord=central_coordinates_ngc1097, cutout_size=cutout_size_ngc1097)
cutout_alma_ngc1566 = hf.get_img_cutout(img=alma_hdu_ngc1566[0].data, wcs=WCS(alma_hdu_ngc1566[0].header),
                                        coord=central_coordinates_ngc1566, cutout_size=cutout_size_ngc1566)



# get color color data
color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
ra_ml = np.load('../color_color/data_output/ra_ml.npy')
dec_ml = np.load('../color_color/data_output/dec_ml.npy')
age_ml = np.load('../color_color/data_output/age_ml.npy')
target_name_ml = np.load('../color_color/data_output/target_name_ml.npy')
clcl_color_ml = np.load('../color_color/data_output/clcl_color_ml.npy')

# categorize data
# get groups of the hull

convex_hull_gc_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_gc_ml_12.fits')[1].data
convex_hull_cascade_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_cascade_ml_12.fits')[1].data
convex_hull_young_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_young_ml_12.fits')[1].data

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

# get position
mask_ngc0628_ml = target_name_ml == 'ngc0628c'
pos_ngc0628_ml = cutout_alma_ngc0628.wcs.world_to_pixel(SkyCoord(ra=ra_ml[mask_ngc0628_ml]*u.deg,
                                                                 dec=dec_ml[mask_ngc0628_ml]*u.deg))
x_ngc0628_ml, y_ngc0628_ml = pos_ngc0628_ml[1], pos_ngc0628_ml[0]
gc_ngc0628_ml = in_hull_gc_ml[mask_ngc0628_ml] * (age_ml[mask_ngc0628_ml] > 10)
cascade_ngc0628_ml = in_hull_cascade_ml[mask_ngc0628_ml] * (age_ml[mask_ngc0628_ml] > 10)
young_ngc0628_ml = in_hull_young_ml[mask_ngc0628_ml] + (age_ml[mask_ngc0628_ml] < 10)

mask_ngc1097_ml = target_name_ml == 'ngc1097'
pos_ngc1097_ml = cutout_alma_ngc1097.wcs.world_to_pixel(SkyCoord(ra=ra_ml[mask_ngc1097_ml]*u.deg,
                                                                 dec=dec_ml[mask_ngc1097_ml]*u.deg))
x_ngc1097_ml, y_ngc1097_ml = pos_ngc1097_ml[1], pos_ngc1097_ml[0]
gc_ngc1097_ml = in_hull_gc_ml[mask_ngc1097_ml] * (age_ml[mask_ngc1097_ml] > 10)
cascade_ngc1097_ml = in_hull_cascade_ml[mask_ngc1097_ml] * (age_ml[mask_ngc1097_ml] > 10)
young_ngc1097_ml = in_hull_young_ml[mask_ngc1097_ml] + (age_ml[mask_ngc1097_ml] < 10)

mask_ngc1566_ml = target_name_ml == 'ngc1566'
pos_ngc1566_ml = cutout_alma_ngc1566.wcs.world_to_pixel(SkyCoord(ra=ra_ml[mask_ngc1566_ml]*u.deg,
                                                                 dec=dec_ml[mask_ngc1566_ml]*u.deg))
x_ngc1566_ml, y_ngc1566_ml = pos_ngc1566_ml[1], pos_ngc1566_ml[0]
gc_ngc1566_ml = in_hull_gc_ml[mask_ngc1566_ml] * (age_ml[mask_ngc1566_ml] > 10)
cascade_ngc1566_ml = in_hull_cascade_ml[mask_ngc1566_ml] * (age_ml[mask_ngc1566_ml] > 10)
young_ngc1566_ml = in_hull_young_ml[mask_ngc1566_ml] + (age_ml[mask_ngc1566_ml] < 10)


kernal_size_prop = 0.02
shape_ngc0628 = cutout_alma_ngc0628.data.shape
shape_ngc1097 = cutout_alma_ngc1097.data.shape
shape_ngc1566 = cutout_alma_ngc1566.data.shape

print('shape_ngc0628 ', shape_ngc0628)
print('shape_ngc1097 ', shape_ngc1097)
print('shape_ngc1566 ', shape_ngc1566)


gc_image_ngc0628_ml, cascade_image_ngc0628_ml, young_image_ngc0628_ml, rgb_image_ngc0628_ml = (
    get_rgb_images(kernal_size_prop=kernal_size_prop, shape=shape_ngc0628, x=x_ngc0628_ml, y=y_ngc0628_ml,
                   mask_r=gc_ngc0628_ml, mask_g=cascade_ngc0628_ml, mask_b=young_ngc0628_ml))

gc_image_ngc1097_ml, cascade_image_ngc1097_ml, young_image_ngc1097_ml, rgb_image_ngc1097_ml = (
    get_rgb_images(kernal_size_prop=kernal_size_prop, shape=shape_ngc1097, x=x_ngc1097_ml, y=y_ngc1097_ml,
                   mask_r=gc_ngc1097_ml, mask_g=cascade_ngc1097_ml, mask_b=young_ngc1097_ml))

gc_image_ngc1566_ml, cascade_image_ngc1566_ml, young_image_ngc1566_ml, rgb_image_ngc1566_ml = (
    get_rgb_images(kernal_size_prop=kernal_size_prop, shape=shape_ngc1566, x=x_ngc1566_ml, y=y_ngc1566_ml,
                   mask_r=gc_ngc1566_ml, mask_g=cascade_ngc1566_ml, mask_b=young_ngc1566_ml))


figure = plt.figure(figsize=(48, 35))
fontsize = 40

ax_img_gc_ngc0628 = figure.add_axes([-0.11, 0.68, 0.5, 0.3], projection=cutout_alma_ngc0628.wcs)
ax_img_cascade_ngc0628 = figure.add_axes([0.135, 0.68, 0.5, 0.3], projection=cutout_alma_ngc0628.wcs)
ax_img_young_ngc0628 = figure.add_axes([0.38, 0.68, 0.5, 0.3], projection=cutout_alma_ngc0628.wcs)
ax_img_rgb_ngc0628 = figure.add_axes([0.625, 0.68, 0.5, 0.3], projection=cutout_alma_ngc0628.wcs)

ax_img_gc_ngc1097 = figure.add_axes([-0.11, 0.355, 0.5, 0.3], projection=cutout_alma_ngc1097.wcs)
ax_img_cascade_ngc1097 = figure.add_axes([0.135, 0.355, 0.5, 0.3], projection=cutout_alma_ngc1097.wcs)
ax_img_young_ngc1097 = figure.add_axes([0.38, 0.355, 0.5, 0.3], projection=cutout_alma_ngc1097.wcs)
ax_img_rgb_ngc1097 = figure.add_axes([0.625, 0.355, 0.5, 0.3], projection=cutout_alma_ngc1097.wcs)


ax_img_gc_ngc1566 = figure.add_axes([-0.11, 0.03, 0.5, 0.3], projection=cutout_alma_ngc1566.wcs)
ax_img_cascade_ngc1566 = figure.add_axes([0.135, 0.03, 0.5, 0.3], projection=cutout_alma_ngc1566.wcs)
ax_img_young_ngc1566 = figure.add_axes([0.38, 0.03, 0.5, 0.3], projection=cutout_alma_ngc1566.wcs)
ax_img_rgb_ngc1566 = figure.add_axes([0.625, 0.03, 0.5, 0.3], projection=cutout_alma_ngc1566.wcs)


plot_panels(ax_img_r=ax_img_gc_ngc0628, ax_img_g=ax_img_cascade_ngc0628, ax_img_b=ax_img_young_ngc0628,
            ax_img_rgb=ax_img_rgb_ngc0628,
            img_r=gc_image_ngc0628_ml, img_g=cascade_image_ngc0628_ml, img_b=young_image_ngc0628_ml,
            img_rgb=rgb_image_ngc0628_ml,
            contour_data=cutout_alma_ngc0628.data, contour_color='gold', alpha=1.0, linewidths=2)

plot_panels(ax_img_r=ax_img_gc_ngc1097, ax_img_g=ax_img_cascade_ngc1097, ax_img_b=ax_img_young_ngc1097,
            ax_img_rgb=ax_img_rgb_ngc1097,
            img_r=gc_image_ngc1097_ml, img_g=cascade_image_ngc1097_ml, img_b=young_image_ngc1097_ml,
            img_rgb=rgb_image_ngc1097_ml,
            contour_data=cutout_alma_ngc1097.data, contour_color='gold', alpha=1.0, linewidths=2)

plot_panels(ax_img_r=ax_img_gc_ngc1566, ax_img_g=ax_img_cascade_ngc1566, ax_img_b=ax_img_young_ngc1566,
            ax_img_rgb=ax_img_rgb_ngc1566,
            img_r=gc_image_ngc1566_ml, img_g=cascade_image_ngc1566_ml, img_b=young_image_ngc1566_ml,
            img_rgb=rgb_image_ngc1566_ml,
            contour_data=cutout_alma_ngc1566.data, contour_color='gold', alpha=1.0, linewidths=2)

ax_img_gc_ngc0628.set_title('Old Globular Clusters (ML)', fontsize=fontsize)
ax_img_cascade_ngc0628.set_title('Middle Aged Clusters (ML)', fontsize=fontsize)
ax_img_young_ngc0628.set_title('Young Clusters (ML) + ALMA', fontsize=fontsize)
ax_img_rgb_ngc0628.set_title('RGB (ML)', fontsize=fontsize)

ax_img_gc_ngc0628.text(shape_ngc0628[0]*0.05, shape_ngc0628[1]*0.95, 'NGC 628',
                       horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
ax_img_gc_ngc1097.text(shape_ngc1097[0]*0.05, shape_ngc1097[1]*0.95, 'NGC 1097',
                       horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
ax_img_gc_ngc1566.text(shape_ngc1566[0]*0.05, shape_ngc1566[1]*0.95, 'NGC 1566',
                       horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)



plotting_tools.arr_axis_params(ax=ax_img_gc_ngc0628, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_cascade_ngc0628, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_young_ngc0628, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_rgb_ngc0628, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)


plotting_tools.arr_axis_params(ax=ax_img_gc_ngc1097, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_cascade_ngc1097, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_young_ngc1097, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_rgb_ngc1097, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)



plotting_tools.arr_axis_params(ax=ax_img_gc_ngc1566, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=-0.3, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_cascade_ngc1566, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_young_ngc1566, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_rgb_ngc1566, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)


plt.savefig('plot_output/spatial_dist_panel.png')
plt.savefig('plot_output/spatial_dist_panel.pdf')


exit()






