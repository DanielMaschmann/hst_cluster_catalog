import numpy as np
from photometry_tools import helper_func as hf, plotting_tools
from photometry_tools.data_access import CatalogAccess
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
import multicolorfits as mcf

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.spatial import ConvexHull
from matplotlib import patheffects


# get color-color data
color_vi_hum = np.load('../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../color_color/data_output/color_ub_hum.npy')
detect_u_hum = np.load('../color_color/data_output/detect_u_hum.npy')
detect_b_hum = np.load('../color_color/data_output/detect_b_hum.npy')
detect_v_hum = np.load('../color_color/data_output/detect_v_hum.npy')
detect_i_hum = np.load('../color_color/data_output/detect_i_hum.npy')
x_hum = np.load('../color_color/data_output/x_hum.npy')
y_hum = np.load('../color_color/data_output/y_hum.npy')
ra_hum = np.load('../color_color/data_output/ra_hum.npy')
dec_hum = np.load('../color_color/data_output/dec_hum.npy')
age_hum = np.load('../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../color_color/data_output/ebv_hum.npy')
target_name_hum = np.load('../color_color/data_output/target_name_hum.npy')
clcl_color_hum = np.load('../color_color/data_output/clcl_color_hum.npy')

color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
detect_u_ml = np.load('../color_color/data_output/detect_u_ml.npy')
detect_b_ml = np.load('../color_color/data_output/detect_b_ml.npy')
detect_v_ml = np.load('../color_color/data_output/detect_v_ml.npy')
detect_i_ml = np.load('../color_color/data_output/detect_i_ml.npy')
x_ml = np.load('../color_color/data_output/x_ml.npy')
y_ml = np.load('../color_color/data_output/y_ml.npy')
ra_ml = np.load('../color_color/data_output/ra_ml.npy')
dec_ml = np.load('../color_color/data_output/dec_ml.npy')
age_ml = np.load('../color_color/data_output/age_ml.npy')
ebv_ml = np.load('../color_color/data_output/ebv_ml.npy')
target_name_ml = np.load('../color_color/data_output/target_name_ml.npy')
clcl_color_ml = np.load('../color_color/data_output/clcl_color_ml.npy')
detectable_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml


model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')


# categorize data
# get groups of the hull
# convex_hull_gc_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_gc_ml_12.fits')[1].data
# convex_hull_cascade_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_cascade_ml_12.fits')[1].data
# convex_hull_young_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_young_ml_12.fits')[1].data


vi_hull_ogc_ubvi_hum_1 = np.load('../segmentation/data_output/vi_hull_ogc_ubvi_hum_1.npy')
ub_hull_ogc_ubvi_hum_1 = np.load('../segmentation/data_output/ub_hull_ogc_ubvi_hum_1.npy')
vi_hull_ogc_ubvi_ml_1 = np.load('../segmentation/data_output/vi_hull_ogc_ubvi_ml_1.npy')
ub_hull_ogc_ubvi_ml_1 = np.load('../segmentation/data_output/ub_hull_ogc_ubvi_ml_1.npy')

vi_hull_mid_ubvi_hum_1 = np.load('../segmentation/data_output/vi_hull_mid_ubvi_hum_1.npy')
ub_hull_mid_ubvi_hum_1 = np.load('../segmentation/data_output/ub_hull_mid_ubvi_hum_1.npy')
vi_hull_mid_ubvi_ml_1 = np.load('../segmentation/data_output/vi_hull_mid_ubvi_ml_1.npy')
ub_hull_mid_ubvi_ml_1 = np.load('../segmentation/data_output/ub_hull_mid_ubvi_ml_1.npy')

vi_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/vi_hull_young_ubvi_hum_3.npy')
ub_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/ub_hull_young_ubvi_hum_3.npy')
vi_hull_young_ubvi_ml_3 = np.load('../segmentation/data_output/vi_hull_young_ubvi_ml_3.npy')
ub_hull_young_ubvi_ml_3 = np.load('../segmentation/data_output/ub_hull_young_ubvi_ml_3.npy')


convex_hull_vi_gc_ml = vi_hull_ogc_ubvi_ml_1
convex_hull_ub_gc_ml = ub_hull_ogc_ubvi_ml_1
convex_hull_vi_cascade_ml = vi_hull_mid_ubvi_ml_1
convex_hull_ub_cascade_ml = ub_hull_mid_ubvi_ml_1
convex_hull_vi_young_ml = vi_hull_young_ubvi_ml_3
convex_hull_ub_young_ml = ub_hull_young_ubvi_ml_3

hull_gc_ml = ConvexHull(np.array([convex_hull_vi_gc_ml, convex_hull_ub_gc_ml]).T)
hull_cascade_ml = ConvexHull(np.array([convex_hull_vi_cascade_ml, convex_hull_ub_cascade_ml]).T)
hull_young_ml = ConvexHull(np.array([convex_hull_vi_young_ml, convex_hull_ub_young_ml]).T)

in_hull_gc_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_gc_ml)
in_hull_cascade_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_cascade_ml)
in_hull_young_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_young_ml)


red_color = '#FF2400'
green_color = '#0FFF50'
blue_color = '#0096FF'

red_color_hst = '#FF4433'
green_color_hst = '#0FFF50'
blue_color_hst = '#1F51FF'

def get_image_data(galaxy_name, n_bins=250, add_side_pixel_frac=0.03, kernel_size_prop=0.01):

    if galaxy_name == 'ngc0628':
        target_mask = (target_name_ml == 'ngc0628e') | (target_name_ml == 'ngc0628c')
    else:
        target_mask = target_name_ml == galaxy_name

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

    young_ml = age_ml < 10

    mask_gc_ml = in_hull_gc_ml * detectable_ml * np.invert(young_ml) * target_mask
    mask_cascade_ml = in_hull_cascade_ml * detectable_ml * np.invert(young_ml) * target_mask
    mask_young_ml = ((detectable_ml * young_ml * (in_hull_gc_ml + in_hull_cascade_ml)) +
                      (in_hull_young_ml * detectable_ml * np.invert(in_hull_cascade_ml))) * target_mask

    # mask_gc_ml = target_mask * in_hull_gc_ml * (age_ml >= 10)
    # mask_cascade_ml = target_mask * in_hull_cascade_ml * (age_ml >= 10)
    # mask_young_ml = target_mask * in_hull_young_ml + ((age_ml < 10) * target_mask)
    # create histogram
    hist_gc = np.histogram2d(x_ml[mask_gc_ml], y_ml[mask_gc_ml], bins=(x_bins, y_bins))[0].T
    hist_cascade = np.histogram2d(x_ml[mask_cascade_ml], y_ml[mask_cascade_ml], bins=(x_bins, y_bins))[0].T
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
    kernel_size = int(hist_young.shape[0] * kernel_size_prop) * 3
    if (kernel_size % 2) == 0:
        kernel_size += 1
    kernel_fwhm = hist_young.shape[0] * kernel_size_prop
    kernel = make_2dgaussian_kernel(kernel_fwhm, size=kernel_size)
    hist_gc = convolve(hist_gc, kernel)
    hist_cascade = convolve(hist_cascade, kernel)
    hist_young = convolve(hist_young, kernel)
    print('kernel_fwhm ', kernel_fwhm)
    # hist_gc /= np.sum(hist_gc)
    # hist_cascade /= np.sum(hist_cascade)
    # hist_young /= np.sum(hist_young)

    grey_gc = mcf.greyRGBize_image(hist_gc, rescalefn='asinh', scaletype='abs', min_max=[0.0, 1/kernel_fwhm],
                                      gamma=4.0, checkscale=False)
    grey_cascade = mcf.greyRGBize_image(hist_cascade, rescalefn='asinh', scaletype='abs', min_max=[0.0, 1/kernel_fwhm],
                                      gamma=4.0, checkscale=False)
    grey_young = mcf.greyRGBize_image(hist_young, rescalefn='asinh', scaletype='abs', min_max=[0.0, 1/kernel_fwhm],
                                      gamma=4.0, checkscale=False)
    r = mcf.colorize_image(grey_gc, red_color, colorintype='hex', gammacorr_color=2.8)
    g = mcf.colorize_image(grey_cascade, green_color, colorintype='hex', gammacorr_color=2.8)
    b = mcf.colorize_image(grey_young, blue_color, colorintype='hex', gammacorr_color=2.8)

    gc_image = mcf.combine_multicolor([r], gamma=2.8, inverse=False)
    cascade_image = mcf.combine_multicolor([g], gamma=2.8, inverse=False)
    young_image = mcf.combine_multicolor([b], gamma=2.8, inverse=False)
    rgb_image = mcf.combine_multicolor([r, g, b], gamma=2.8, inverse=False)

    # make background white
    # mask_0_gc_image = (gc_image[:,:, 0] < 0.2) & (gc_image[:,:, 1] < 0.2) & (gc_image[:,:, 2] < 0.2)
    # mask_0_cascade_image = (cascade_image[:,:, 0] < 0.2) & (cascade_image[:,:, 1] < 0.2) & (cascade_image[:,:, 2] < 0.2)
    # mask_0_youncascade_image = (young_image[:,:, 0] < 0.2) & (young_image[:,:, 1] < 0.2) & (young_image[:,:, 2] < 0.2)
    # mask_0_rgb_image = (rgb_image[:,:, 0] < 0.2) & (rgb_image[:,:, 1] < 0.2) & (rgb_image[:,:, 2] < 0.2)

    # print('np.sum(hist_gc) ', np.sum(hist_gc))
    # print('np.sum(cascade_image) ', np.sum(hist_cascade))
    # print('np.sum(young_image) ', np.sum(hist_young))
    # print('max gc_image[:, :, 0] ', np.max(gc_image[:, :, 0]))
    # print('max gc_image[:, :, 1] ', np.max(gc_image[:, :, 1]))
    # print('max gc_image[:, :, 2] ', np.max(gc_image[:, :, 2]))
    # print('max cascade_image[:, :, 0] ', np.max(cascade_image[:, :, 0]))
    # print('max cascade_image[:, :, 1] ', np.max(cascade_image[:, :, 1]))
    # print('max cascade_image[:, :, 2] ', np.max(cascade_image[:, :, 2]))
    # print('max young_image[:, :, 0] ', np.max(young_image[:, :, 0]))
    # print('max young_image[:, :, 1] ', np.max(young_image[:, :, 1]))
    # print('max young_image[:, :, 2] ', np.max(young_image[:, :, 2]))

    mask_0_gc_image = hist_gc < 0.01
    mask_0_cascade_image = hist_cascade < 0.01
    mask_0_youncascade_image = hist_young < 0.01

    # mask_0_gc_image = ((gc_image[:, :, 0] < np.sum(hist_gc) / 5) &
    #                     (gc_image[:, :, 1] < np.sum(hist_gc) / 5) &
    #                     (gc_image[:, :, 2] < np.sum(hist_gc) / 5))
    # mask_0_cascade_image = ((cascade_image[:, :, 0] < 15 / np.sum(hist_cascade)) &
    #                     (cascade_image[:, :, 1] < 15 / np.sum(hist_cascade)) &
    #                     (cascade_image[:, :, 2] < 15 / np.sum(hist_cascade)))
    # mask_0_youncascade_image = ((young_image[:, :, 0] < 15 / np.sum(hist_young)) &
    #                     (young_image[:, :, 1] < 15 / np.sum(hist_young)) &
    #                     (young_image[:, :, 2] < 15 / np.sum(hist_young)))
    #

    mask_0_rgb_image = ((rgb_image[:, :, 0] < 15 / np.sum(hist_gc)) &
                        (rgb_image[:, :, 1] < 15 / np.sum(hist_cascade)) &
                        (rgb_image[:, :, 2] < 15 / np.sum(hist_young)))

    gc_image[mask_0_gc_image] = 1.0
    cascade_image[mask_0_cascade_image] = 1.0
    young_image[mask_0_youncascade_image] = 1.0
    rgb_image[mask_0_rgb_image] = 1.0


    # gte alma observation data
    alma_hdu = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_mom0.fits' %
                         (galaxy_name, galaxy_name))
    alma_wcs = WCS(alma_hdu[0].header)

    alma_data_reprojected = plotting_tools.reproject_image(data=alma_hdu[0].data, wcs=alma_wcs,
                                                           new_wcs=wcs_hist, new_shape=young_image[:,:,0].shape)
    alma_borders_reprojected = np.invert(np.isnan(alma_data_reprojected))


    hdu_rgb_img_hst = fits.open('/home/benutzer/data/PHANGS-HST/rgb_img/%s_BVI_RGB.fits' % (galaxy_name))
    rgb_img_hst_data_r = plotting_tools.reproject_image(data=hdu_rgb_img_hst[0].data[0],
                                                            wcs=WCS(hdu_rgb_img_hst[0].header, naxis=2),
                                                            new_wcs=wcs_hist, new_shape=young_image[:,:,0].shape)
    rgb_img_hst_data_g = plotting_tools.reproject_image(data=hdu_rgb_img_hst[0].data[1],
                                                            wcs=WCS(hdu_rgb_img_hst[0].header, naxis=2),
                                                            new_wcs=wcs_hist, new_shape=young_image[:,:,0].shape)
    rgb_img_hst_data_b = plotting_tools.reproject_image(data=hdu_rgb_img_hst[0].data[2],
                                                            wcs=WCS(hdu_rgb_img_hst[0].header, naxis=2),
                                                            new_wcs=wcs_hist, new_shape=young_image[:,:,0].shape)
    rgb_img_hst_data = np.array([rgb_img_hst_data_r.T, rgb_img_hst_data_g.T, rgb_img_hst_data_b.T])

    grey_r = mcf.greyRGBize_image(rgb_img_hst_data.T[:,:,0], rescalefn='asinh', scaletype='perc', min_max=[0.3, 99.5],
                                      gamma=17.5, checkscale=False)
    grey_g = mcf.greyRGBize_image(rgb_img_hst_data.T[:,:,1], rescalefn='asinh', scaletype='perc', min_max=[0.3, 99.5],
                                      gamma=17.5, checkscale=False)
    grey_b = mcf.greyRGBize_image(rgb_img_hst_data.T[:,:,2], rescalefn='asinh', scaletype='perc', min_max=[0.3, 99.5],
                                      gamma=17.5, checkscale=False)
    r = mcf.colorize_image(grey_r, red_color_hst, colorintype='hex', gammacorr_color=17.5)
    g = mcf.colorize_image(grey_g, green_color_hst, colorintype='hex', gammacorr_color=17.5)
    b = mcf.colorize_image(grey_b, blue_color_hst, colorintype='hex', gammacorr_color=17.5)
    hst_rgb_image = mcf.combine_multicolor([r, g, b], gamma=17.5, inverse=False)


    return alma_data_reprojected, alma_borders_reprojected, hst_rgb_image, gc_image, cascade_image, young_image, rgb_image, wcs_hist


def plot_panels(ax_img_r, ax_img_g, ax_img_b, ax_img_hst, img_r, img_g, img_b, img_hst,
                contour_data, contour_borders, plot_contours=True, contour_color='gold', alpha=1.0, linewidths=4):
    ax_img_r.imshow(img_r)
    ax_img_g.imshow(img_g)
    ax_img_b.imshow(img_b)
    #ax_img_rgb.imshow(img_rgb)
    ax_img_hst.imshow(img_hst)

    # bkg_estimator = MedianBackground()
    # sigma_clip = SigmaClip(sigma=3.0)
    # contour_data[np.isnan(contour_data)] = 0
    # bkg = Background2D(contour_data, (50, 50), filter_size=(3, 3),
    #                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    # contour_data -= bkg.background

    # mean, median, std = sigma_clipped_stats(contour_data, sigma=3.0)
    # levels_std = np.array(np.arange(start=1, stop=5, step=1), dtype=float)
    # levels = np.ones(len(levels_std)) * mean
    # levels += levels_std
    # # print(levels)
    # print(np.nanmax(contour_data))
    # print('mean, median, std ', mean, median, std)
    # ax_img_b.contour(contour_data, levels=levels[:-2], colors=contour_color, alpha=alpha, linewidths=linewidths)
    # levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    # levels = [0.3, 0.5, 0.7, 0.9, 0.99]
    if plot_contours:
        levels = [np.nanpercentile(contour_data, 90),
                  np.nanpercentile(contour_data, 95),
                  np.nanpercentile(contour_data, 99)]
        print(levels)
        ax_img_b.contour(contour_data, levels=levels, colors=contour_color, alpha=alpha, linewidths=linewidths)

    ax_img_b.contour(contour_borders, levels=0, colors='k', linewidths=4)


def plot_names(ax_img_r, ax_img_g, ax_img_b, ax_img_hst, shape, target, delta_ms, fontsize, plot_identifier=True):

    if (target[0:3] == 'ngc') & (target[3] == '0'):
        target_name_str = target[0:3] + ' ' + target[4:]
    elif target[0:2] == 'ic':
        target_name_str = target[0:2] + ' ' + target[2:]
    elif target[0:3] == 'ngc':
        target_name_str = target[0:3] + ' ' + target[3:]
    else:
        target_name_str = target
    target_name_str = target_name_str.upper() + r'  $\Delta$MS=%.1f dex' % delta_ms
    ax_img_r.set_title(target_name_str, fontsize=fontsize + 15)

    if not plot_identifier:
        return None

    pe = [patheffects.withStroke(linewidth=3, foreground="w")]

    ax_img_r.text(0.03, 0.97, 'Old Globular Cluster Clump', horizontalalignment='left', verticalalignment='top',
                  color='tab:red', fontsize=fontsize, transform=ax_img_r.transAxes, path_effects=pe)
    ax_img_g.text(0.03, 0.97, 'Middle-Aged Plume', horizontalalignment='left', verticalalignment='top',
                  color='tab:green', fontsize=fontsize, transform=ax_img_g.transAxes, path_effects=pe)
    ax_img_b.text(0.03, 0.97, 'Young Cluster Locus', horizontalalignment='left', verticalalignment='top',
                  color='tab:blue', fontsize=fontsize, transform=ax_img_b.transAxes, path_effects=pe)
    ax_img_b.text(0.03, 0.90, '+ ALMA', horizontalalignment='left', verticalalignment='top',
                  color='magenta', fontsize=fontsize, transform=ax_img_b.transAxes, path_effects=pe)

    ax_img_r.text(0.97, 0.03, 'ML', horizontalalignment='right', verticalalignment='bottom',
                  color='k', fontsize=fontsize, transform=ax_img_r.transAxes, path_effects=pe)
    ax_img_g.text(0.97, 0.03, 'ML', horizontalalignment='right', verticalalignment='bottom',
                  color='k', fontsize=fontsize, transform=ax_img_g.transAxes, path_effects=pe)
    ax_img_b.text(0.97, 0.03, 'ML', horizontalalignment='right', verticalalignment='bottom',
                  color='k', fontsize=fontsize, transform=ax_img_b.transAxes, path_effects=pe)


    ax_img_hst.text(0.03, 0.97, 'HST', horizontalalignment='left', verticalalignment='top',
                    color='white', fontsize=fontsize, transform=ax_img_hst.transAxes, path_effects=pe)
    ax_img_hst.text(0.18, 0.97, 'B', horizontalalignment='left', verticalalignment='top',
                    color='blue', fontsize=fontsize, transform=ax_img_hst.transAxes, path_effects=pe)
    ax_img_hst.text(0.23, 0.97, 'V', horizontalalignment='left', verticalalignment='top',
                    color='green', fontsize=fontsize, transform=ax_img_hst.transAxes, path_effects=pe)
    ax_img_hst.text(0.28, 0.97, 'I', horizontalalignment='left', verticalalignment='top',
                    color='red', fontsize=fontsize, transform=ax_img_hst.transAxes, path_effects=pe)


    #
    # ax_img_r.text(shape[0]*0.04, shape[1]*0.96, 'Old Globular Clusters (ML)', color='tab:red',
    #               horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 3)
    # ax_img_g.text(shape[0]*0.04, shape[1]*0.96, 'Middle-Aged Plume (ML)', color='tab:green',
    #               horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 3)
    #
    # ax_img_b.text(shape[0]*0.04, shape[1]*0.96, r'Young Cluster Locus (ML) ', color='tab:blue',
    #               horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 3)
    # ax_img_b.text(shape[0]*0.04, shape[1]*0.88, r' + ALMA ', color='k',
    #               horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 3)
    #
    # ax_img_hst.text(shape[0]*0.05, shape[1]*0.95, 'HST', color='white',
    #               horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
    # ax_img_hst.text(shape[0]*0.21, shape[1]*0.95, 'B', color='blue',
    #               horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
    # ax_img_hst.text(shape[0]*0.26, shape[1]*0.95, 'V', color='green',
    #               horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
    # ax_img_hst.text(shape[0]*0.31, shape[1]*0.95, 'I', color='red',
    #               horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)

sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = CatalogAccess(sample_table_path=sample_table_path)
catalog_access.load_sample_table()


n_bins = 250
add_side_pixel_frac = 3/100
kernel_size_prop = 0.01


galaxy_name_list = ['ic1954', 'ic5332', 'ngc0628',
                     'ngc0685', 'ngc1087', 'ngc1097',
                     'ngc1300', 'ngc1317', 'ngc1365',
                     'ngc1385', 'ngc1433', 'ngc1512',
                     'ngc1559', 'ngc1566', 'ngc1672',
                     'ngc1792', 'ngc2775', 'ngc2835',
                     'ngc2903', 'ngc3351', 'ngc3621',
                     'ngc3627', 'ngc4254', 'ngc4298',
                     'ngc4303', 'ngc4321', 'ngc4535',
                     'ngc4536', 'ngc4548', 'ngc4569',
                     'ngc4571', 'ngc4654', 'ngc4689',
                     'ngc4826', 'ngc5068', 'ngc5248',
                     'ngc6744', 'ngc7496']

list_delta_ms = np.zeros(len(galaxy_name_list))

for index, target in enumerate(galaxy_name_list):
    list_delta_ms[index] = catalog_access.get_target_delta_ms(target=target)


sort = np.argsort(list_delta_ms)[::-1]
galaxy_name_list = np.array(galaxy_name_list)[sort]
list_delta_ms = np.array(list_delta_ms)[sort]


for index in range(10):
    target_1 = galaxy_name_list[index * 4]
    target_2 = galaxy_name_list[index * 4 + 1]
    delta_ms_1 = list_delta_ms[index * 4]
    delta_ms_2 = list_delta_ms[index * 4 + 1]
    if index == 9:
        target_3 = None
        target_4 = None
        delta_ms_3 = None
        delta_ms_4 = None
    else:
        target_3 = galaxy_name_list[index * 4 + 2]
        target_4 = galaxy_name_list[index * 4 + 3]
        delta_ms_3 = list_delta_ms[index * 4 + 2]
        delta_ms_4 = list_delta_ms[index * 4 + 3]


    alma_data_1, alma_borders_1, rgb_img_hst_img_1, gc_img_1, cascade_img_1, young_img_1, rgb_img_1, wcs_hist_1 = (
        get_image_data(galaxy_name=target_1, n_bins=n_bins, add_side_pixel_frac=add_side_pixel_frac,
                       kernel_size_prop=kernel_size_prop))
    alma_data_2,  alma_borders_2, rgb_img_hst_img_2, gc_img_2, cascade_img_2, young_img_2, rgb_img_2, wcs_hist_2 = (
        get_image_data(galaxy_name=target_2, n_bins=n_bins, add_side_pixel_frac=add_side_pixel_frac,
                       kernel_size_prop=kernel_size_prop))
    if index != 9:
        alma_data_3, alma_borders_3, rgb_img_hst_img_3, gc_img_3, cascade_img_3, young_img_3, rgb_img_3, wcs_hist_3 = (
            get_image_data(galaxy_name=target_3, n_bins=n_bins, add_side_pixel_frac=add_side_pixel_frac,
                           kernel_size_prop=kernel_size_prop))
        alma_data_4, alma_borders_4, rgb_img_hst_img_4, gc_img_4, cascade_img_4, young_img_4, rgb_img_4, wcs_hist_4 = (
            get_image_data(galaxy_name=target_4, n_bins=n_bins, add_side_pixel_frac=add_side_pixel_frac,
                           kernel_size_prop=kernel_size_prop))


    figure = plt.figure(figsize=(50, 50))

    fontsize = 50

    ax_size = 0.225

    ax_horizont1 = 0.75
    ax_horizont2 = 0.51
    ax_horizont3 = 0.27
    ax_horizont4 = 0.03
    # ax_horizont5 = 0.005

    ax_vertical1 = 0.035
    ax_vertical2 = 0.28
    ax_vertical3 = 0.525
    ax_vertical4 = 0.77

    ax_img_gc_1 = figure.add_axes([ax_vertical1, ax_horizont1, ax_size, ax_size], projection=wcs_hist_1)
    ax_img_cascade_1 = figure.add_axes([ax_vertical1, ax_horizont2, ax_size, ax_size], projection=wcs_hist_1)
    ax_img_young_1 = figure.add_axes([ax_vertical1, ax_horizont3, ax_size, ax_size], projection=wcs_hist_1)
    #ax_img_rgb_1 = figure.add_axes([ax_vertical1, ax_horizont4, ax_size, ax_size], projection=wcs_hist_1)
    ax_img_hst_1 = figure.add_axes([ax_vertical1, ax_horizont4, ax_size, ax_size], projection=wcs_hist_1)

    ax_img_gc_2 = figure.add_axes([ax_vertical2, ax_horizont1, ax_size, ax_size], projection=wcs_hist_2)
    ax_img_cascade_2 = figure.add_axes([ax_vertical2, ax_horizont2, ax_size, ax_size], projection=wcs_hist_2)
    ax_img_young_2 = figure.add_axes([ax_vertical2, ax_horizont3, ax_size, ax_size], projection=wcs_hist_2)
    #ax_img_rgb_2 = figure.add_axes([ax_vertical2, ax_horizont4, ax_size, ax_size], projection=wcs_hist_2)
    ax_img_hst_2 = figure.add_axes([ax_vertical2, ax_horizont4, ax_size, ax_size], projection=wcs_hist_2)

    if index != 9:

        ax_img_gc_3 = figure.add_axes([ax_vertical3, ax_horizont1, ax_size, ax_size], projection=wcs_hist_3)
        ax_img_cascade_3 = figure.add_axes([ax_vertical3, ax_horizont2, ax_size, ax_size], projection=wcs_hist_3)
        ax_img_young_3 = figure.add_axes([ax_vertical3, ax_horizont3, ax_size, ax_size], projection=wcs_hist_3)
        #ax_img_rgb_3 = figure.add_axes([ax_vertical3, ax_horizont4, ax_size, ax_size], projection=wcs_hist_3)
        ax_img_hst_3 = figure.add_axes([ax_vertical3, ax_horizont4, ax_size, ax_size], projection=wcs_hist_3)

        ax_img_gc_4 = figure.add_axes([ax_vertical4, ax_horizont1, ax_size, ax_size], projection=wcs_hist_4)
        ax_img_cascade_4 = figure.add_axes([ax_vertical4, ax_horizont2, ax_size, ax_size], projection=wcs_hist_4)
        ax_img_young_4 = figure.add_axes([ax_vertical4, ax_horizont3, ax_size, ax_size], projection=wcs_hist_4)
        #ax_img_rgb_4 = figure.add_axes([ax_vertical4, ax_horizont4, ax_size, ax_size], projection=wcs_hist_4)
        ax_img_hst_4 = figure.add_axes([ax_vertical4, ax_horizont4, ax_size, ax_size], projection=wcs_hist_4)

    if index == 6:
        plot_contours = False
    else:
        plot_contours = True

    plot_panels(ax_img_r=ax_img_gc_1, ax_img_g=ax_img_cascade_1, ax_img_b=ax_img_young_1,
                ax_img_hst=ax_img_hst_1,
                img_r=gc_img_1, img_g=cascade_img_1, img_b=young_img_1, img_hst=rgb_img_hst_img_1,
                contour_data=alma_data_1, contour_borders=alma_borders_1, plot_contours=plot_contours, contour_color='magenta', alpha=0.7, linewidths=4)

    plot_panels(ax_img_r=ax_img_gc_2, ax_img_g=ax_img_cascade_2, ax_img_b=ax_img_young_2,
                ax_img_hst=ax_img_hst_2,
                img_r=gc_img_2, img_g=cascade_img_2, img_b=young_img_2, img_hst=rgb_img_hst_img_2,
                contour_data=alma_data_2, contour_borders=alma_borders_2, contour_color='magenta', alpha=0.7, linewidths=4)
    if index != 9:
        plot_panels(ax_img_r=ax_img_gc_3, ax_img_g=ax_img_cascade_3, ax_img_b=ax_img_young_3,
                    ax_img_hst=ax_img_hst_3,
                    img_r=gc_img_3, img_g=cascade_img_3, img_b=young_img_3, img_hst=rgb_img_hst_img_3,
                    contour_data=alma_data_3, contour_borders=alma_borders_3, contour_color='magenta', alpha=0.7, linewidths=4)

        plot_panels(ax_img_r=ax_img_gc_4, ax_img_g=ax_img_cascade_4, ax_img_b=ax_img_young_4,
                    ax_img_hst=ax_img_hst_4,
                    img_r=gc_img_4, img_g=cascade_img_4, img_b=young_img_4, img_hst=rgb_img_hst_img_4,
                    contour_data=alma_data_4, contour_borders=alma_borders_4, contour_color='magenta', alpha=0.7, linewidths=4)

    plot_names(ax_img_r=ax_img_gc_1, ax_img_g=ax_img_cascade_1, ax_img_b=ax_img_young_1,
               ax_img_hst=ax_img_hst_1, target=target_1, delta_ms=delta_ms_1, shape=gc_img_1.shape, fontsize=fontsize, plot_identifier=True)
    plot_names(ax_img_r=ax_img_gc_2, ax_img_g=ax_img_cascade_2, ax_img_b=ax_img_young_2,
           ax_img_hst=ax_img_hst_2, target=target_2, delta_ms=delta_ms_2, shape=gc_img_2.shape, fontsize=fontsize, plot_identifier=False)
    if index != 9:
        plot_names(ax_img_r=ax_img_gc_3, ax_img_g=ax_img_cascade_3, ax_img_b=ax_img_young_3,
               ax_img_hst=ax_img_hst_3, target=target_3, delta_ms=delta_ms_3, shape=gc_img_3.shape, fontsize=fontsize, plot_identifier=False)
        plot_names(ax_img_r=ax_img_gc_4, ax_img_g=ax_img_cascade_4, ax_img_b=ax_img_young_4,
                   ax_img_hst=ax_img_hst_4, target=target_4, delta_ms=delta_ms_4, shape=gc_img_4.shape, fontsize=fontsize, plot_identifier=False)

    if index == 0:
        dec_tick_num_1 = 2
    else:
        dec_tick_num_1 = 4

    if (index == 6) | (index == 9):
        dec_tick_num_2 = 2
    else:
        dec_tick_num_2 = 4

    plotting_tools.arr_axis_params(ax=ax_img_gc_1, ra_tick_label=False, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num_1)
    plotting_tools.arr_axis_params(ax=ax_img_cascade_1, ra_tick_label=False, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num_1)
    plotting_tools.arr_axis_params(ax=ax_img_young_1, ra_tick_label=False, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num_1)
    # plotting_tools.arr_axis_params(ax=ax_img_rgb_1, ra_tick_label=False, dec_tick_label=True,
    #                                    ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
    #                                    ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
    #                                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num)
    plotting_tools.arr_axis_params(ax=ax_img_hst_1, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num_1)


    plotting_tools.arr_axis_params(ax=ax_img_gc_2, ra_tick_label=False, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num_2)
    plotting_tools.arr_axis_params(ax=ax_img_cascade_2, ra_tick_label=False, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num_2)
    plotting_tools.arr_axis_params(ax=ax_img_young_2, ra_tick_label=False, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num_2)
    # plotting_tools.arr_axis_params(ax=ax_img_rgb_2, ra_tick_label=False, dec_tick_label=True,
    #                                    ra_axis_label=' ', dec_axis_label=' ',
    #                                    ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
    #                                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num_2)
    plotting_tools.arr_axis_params(ax=ax_img_hst_2, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=dec_tick_num_2)

    if index != 9:
        plotting_tools.arr_axis_params(ax=ax_img_gc_3, ra_tick_label=False, dec_tick_label=True,
                                           ra_axis_label=' ', dec_axis_label=' ',
                                           ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                           fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)
        plotting_tools.arr_axis_params(ax=ax_img_cascade_3, ra_tick_label=False, dec_tick_label=True,
                                           ra_axis_label=' ', dec_axis_label=' ',
                                           ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                           fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)
        plotting_tools.arr_axis_params(ax=ax_img_young_3, ra_tick_label=False, dec_tick_label=True,
                                           ra_axis_label=' ', dec_axis_label=' ',
                                           ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                           fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)
        # plotting_tools.arr_axis_params(ax=ax_img_rgb_3, ra_tick_label=False, dec_tick_label=True,
        #                                    ra_axis_label=' ', dec_axis_label=' ',
        #                                    ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
        #                                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)
        plotting_tools.arr_axis_params(ax=ax_img_hst_3, ra_tick_label=True, dec_tick_label=True,
                                           ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                           ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                           fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)



        plotting_tools.arr_axis_params(ax=ax_img_gc_4, ra_tick_label=False, dec_tick_label=True,
                                           ra_axis_label=' ', dec_axis_label=' ',
                                           ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                           fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)
        plotting_tools.arr_axis_params(ax=ax_img_cascade_4, ra_tick_label=False, dec_tick_label=True,
                                           ra_axis_label=' ', dec_axis_label=' ',
                                           ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                           fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)
        plotting_tools.arr_axis_params(ax=ax_img_young_4, ra_tick_label=False, dec_tick_label=True,
                                           ra_axis_label=' ', dec_axis_label=' ',
                                           ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                                           fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)
        # plotting_tools.arr_axis_params(ax=ax_img_rgb_4, ra_tick_label=False, dec_tick_label=True,
        #                                    ra_axis_label=' ', dec_axis_label=' ',
        #                                    ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
        #                                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)
        plotting_tools.arr_axis_params(ax=ax_img_hst_4, ra_tick_label=True, dec_tick_label=True,
                                           ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                           ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                           fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=4)


    plt.savefig('plot_output/overview_panel_%s.png' % index)
    plt.savefig('plot_output/overview_panel_%s.pdf' % index)
    plt.cla()
    plt.close()
    plt.clf()


