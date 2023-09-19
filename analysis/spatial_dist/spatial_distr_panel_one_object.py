import numpy as np
from photometry_tools import helper_func as hf, plotting_tools
from photometry_tools.analysis_tools import AnalysisTools
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
import multicolorfits as mcf

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.spatial import ConvexHull



# get color color data
color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
x_ml = np.load('../color_color/data_output/x_ml.npy')
y_ml = np.load('../color_color/data_output/y_ml.npy')
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


def create_hist(galaxy_name, n_bins=250, add_side_pixel_frac=0.03, kernel_size_prop=0.02):

    if galaxy_name == 'ngc0628':
        target_mask = (target_name_ml == 'ngc0628e') | (target_name_ml == 'ngc0628c')
    else:
        target_mask = target_name_ml == galaxy_name

    n_added_pix = int(n_bins * add_side_pixel_frac)
    print('n_added_pix ', n_added_pix)

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

    print('x_bin_width ', x_bin_width)
    print('y_bin_width ', y_bin_width)

    center_ra = (left_ra + right_ra) / 2
    center_dec = (lower_dec + upper_dec) / 2
    print('center_ra ', center_ra)
    print('center_dec ', center_dec)

    # ra_bin_width = (right_ra - left_ra) / n_bins
    # dec_bin_width = (upper_dec - lower_dec) / n_bins
    pos_coord_lower_left = SkyCoord(ra=left_ra*u.deg, dec=lower_dec*u.deg)
    pos_coord_lower_right = SkyCoord(ra=right_ra*u.deg, dec=lower_dec*u.deg)
    pos_coord_upper_left = SkyCoord(ra=left_ra*u.deg, dec=upper_dec*u.deg)

    ra_bin_width = (pos_coord_lower_left.separation(pos_coord_lower_right) / n_bins).degree
    dec_bin_width = (pos_coord_lower_left.separation(pos_coord_upper_left) / n_bins).degree

    print('ra_bin_width ', ra_bin_width)
    print('dec_bin_width ', dec_bin_width)

    x_bins = np.linspace(np.nanmin(x_ml[target_mask]) - x_bin_width*n_added_pix,
                         np.nanmax(x_ml[target_mask]) + x_bin_width*n_added_pix, n_bins + 1 + 2*n_added_pix)
    y_bins = np.linspace(np.nanmin(y_ml[target_mask]) - y_bin_width*n_added_pix,
                         np.nanmax(y_ml[target_mask]) + y_bin_width*n_added_pix, n_bins + 1 + 2*n_added_pix)


    mask_gc_ml = target_mask * in_hull_gc_ml * (age_ml >= 10)
    mask_cascade_ml = target_mask * in_hull_cascade_ml * (age_ml >= 10)
    mask_young_ml = target_mask * in_hull_young_ml + ((age_ml < 10) * target_mask)
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

    kernel = make_2dgaussian_kernel(hist_young.shape[0] * kernel_size_prop, size=kernel_size)
    hist_gc = convolve(hist_gc, kernel)
    hist_cascade = convolve(hist_cascade, kernel)
    hist_young = convolve(hist_young, kernel)

    hist_gc /= np.sum(hist_gc)
    hist_cascade /= np.sum(hist_cascade)
    hist_young /= np.sum(hist_young)

    grey_gc = mcf.greyRGBize_image(hist_gc, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                      gamma=4.0, checkscale=False)
    grey_cascade = mcf.greyRGBize_image(hist_cascade, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                      gamma=4.0, checkscale=False)
    grey_young = mcf.greyRGBize_image(hist_young, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                      gamma=4.0, checkscale=False)
    r = mcf.colorize_image(grey_gc, '#FF4433', colorintype='hex', gammacorr_color=2.8)
    g = mcf.colorize_image(grey_cascade, '#0FFF50', colorintype='hex', gammacorr_color=2.8)
    b = mcf.colorize_image(grey_young, '#0096FF', colorintype='hex', gammacorr_color=2.8)

    gc_image = mcf.combine_multicolor([r], gamma=2.8, inverse=False)
    cascade_image = mcf.combine_multicolor([g], gamma=2.8, inverse=False)
    young_image = mcf.combine_multicolor([b], gamma=2.8, inverse=False)
    rgb_image = mcf.combine_multicolor([r, g, b], gamma=2.8, inverse=False)


    # make background white
    mask_0_gc_image = (gc_image[:,:, 0] < 0.05) & (gc_image[:,:, 1] < 0.05) & (gc_image[:,:, 2] < 0.05)
    mask_0_cascade_image = (cascade_image[:,:, 0] < 0.05) & (cascade_image[:,:, 1] < 0.05) & (cascade_image[:,:, 2] < 0.05)
    mask_0_youncascade_image = (young_image[:,:, 0] < 0.05) & (young_image[:,:, 1] < 0.05) & (young_image[:,:, 2] < 0.05)
    mask_0_rgb_image = (rgb_image[:,:, 0] < 0.05) & (rgb_image[:,:, 1] < 0.05) & (rgb_image[:,:, 2] < 0.05)

    gc_image[mask_0_gc_image] = 1.0
    cascade_image[mask_0_cascade_image] = 1.0
    young_image[mask_0_youncascade_image] = 1.0
    rgb_image[mask_0_rgb_image] = 1.0

    return gc_image, cascade_image, young_image, rgb_image, wcs_hist


def plot_panels(ax_img_r, ax_img_g, ax_img_b, ax_img_rgb, img_r, img_g, img_b, img_rgb,
                contour_data, contour_color='gold', alpha=1.0, linewidths=2):
    ax_img_r.imshow(img_r)
    ax_img_g.imshow(img_g)
    ax_img_b.imshow(img_b)
    ax_img_rgb.imshow(img_rgb)
    # from astropy.stats import SigmaClip
    # from photutils.background import Background2D, MedianBackground
    # bkg_estimator = MedianBackground()
    # sigma_clip = SigmaClip(sigma=3.0)
    # print(contour_data.shape)
    # bkg = Background2D(contour_data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    # contour_data = contour_data - bkg.background
    mean, median, std = sigma_clipped_stats(contour_data, sigma=3.0)
    levels_std = np.array(np.arange(start=0.3, stop=5, step=1), dtype=float)
    levels = np.ones(len(levels_std)) * mean
    levels += levels_std
    ax_img_b.contour(contour_data,
                               levels=levels,
                               colors=contour_color, alpha=alpha, linewidths=linewidths)


def get_cc_pos(target):

    if target == 'ngc0628':
        mask_ml = (target_name_ml == 'ngc0628e') | (target_name_ml == 'ngc0628c')
    else:
        mask_ml = target_name_ml == target
    mask_gc_ml = in_hull_gc_ml[mask_ml] * (age_ml[mask_ml] > 10)
    mask_cascade_ml = in_hull_cascade_ml[mask_ml] * (age_ml[mask_ml] > 10)
    mask_young_ml = in_hull_young_ml[mask_ml] + (age_ml[mask_ml] < 10)

    return {'mask_gc_ml': mask_gc_ml, 'mask_cascade_ml': mask_cascade_ml,
            'mask_young_ml': mask_young_ml}



# galaxy_name = 'ngc1566'
n_bins = 250
add_side_pixel_frac = 3/100
kernel_size_prop = 0.02


cut_out_size_dict = {'ic1954': (130, 130), 'ic5332': (120, 120), 'ngc0628': (250, 250),
                     'ngc0685': (250, 250), 'ngc1087': (100, 100), 'ngc1097': (250, 250),
                     'ngc1300': (170, 170), 'ngc1317': (150, 150), 'ngc1365': (120, 120),
                     'ngc1385': (125, 125), 'ngc1433': (170, 170), 'ngc1512': (170, 170),
                     'ngc1559': (250, 250), 'ngc1566': (172.5, 172.5), 'ngc1672': (120, 120),
                     'ngc1792': (200, 200), 'ngc2775': (100, 100), 'ngc2835': (150, 150),
                     'ngc2903': (190, 190), 'ngc3351': (160, 160), 'ngc3621': (200, 200),
                     'ngc3627': (160, 160), 'ngc4254': (170, 170), 'ngc4298': (130, 130),
                     'ngc4303': (160, 160), 'ngc4321': (200, 200), 'ngc4535': (150, 150),
                     'ngc4536': (200, 200), 'ngc4548': (190, 190), 'ngc4569': (160, 160),
                     'ngc4571': (170, 170), 'ngc4654': (200, 200), 'ngc4689': (145, 145),
                     'ngc4826': (150, 150), 'ngc5068': (200, 200), 'ngc5248': (200, 200),
                     'ngc6744': (250, 250), 'ngc7496': (150, 150)
}


for index in range(13):
    print(index)
    target_1 = list(cut_out_size_dict.keys())[index * 3]
    gc_image_1, cascade_image_1, young_image_1, rgb_image_1, wcs_hist_1 = create_hist(galaxy_name=target_1,
                                                                                      n_bins=n_bins,
                                                                                      add_side_pixel_frac=add_side_pixel_frac,
                                                                                      kernel_size_prop=kernel_size_prop)
    cutout_size_1 = cut_out_size_dict[target_1]
    print(young_image_1[:,:,0].shape)
    print(wcs_hist_1)
    alma_hdu_1 = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' % (target_1, target_1))
    alma_data_1 = plotting_tools.reproject_image(data=alma_hdu_1[0].data, wcs=WCS(alma_hdu_1[0].header),
                                                 new_wcs=wcs_hist_1, new_shape=young_image_1[:,:,0].shape)


    target_2 = list(cut_out_size_dict.keys())[index * 3 + 1]
    gc_image_2, cascade_image_2, young_image_2, rgb_image_2, wcs_hist_2 = create_hist(galaxy_name=target_2,
                                                                                      n_bins=n_bins,
                                                                                      add_side_pixel_frac=add_side_pixel_frac,
                                                                                      kernel_size_prop=kernel_size_prop)
    alma_hdu_2 = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' % (target_2, target_2))
    alma_data_2 = plotting_tools.reproject_image(data=alma_hdu_2[0].data, wcs=WCS(alma_hdu_2[0].header),
                                                 new_wcs=wcs_hist_2, new_shape=young_image_2[:,:,0].shape)

    if index == 12:
        target_3 = None
    else:
        target_3 = list(cut_out_size_dict.keys())[index * 3 + 2]
        gc_image_3, cascade_image_3, young_image_3, rgb_image_3, wcs_hist_3 = create_hist(galaxy_name=target_3,
                                                                                  n_bins=n_bins,
                                                                                  add_side_pixel_frac=add_side_pixel_frac,
                                                                                  kernel_size_prop=kernel_size_prop)
        alma_hdu_3 = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' % (target_3, target_3))
        alma_data_3 = plotting_tools.reproject_image(data=alma_hdu_3[0].data, wcs=WCS(alma_hdu_3[0].header),
                                                     new_wcs=wcs_hist_3, new_shape=young_image_3[:,:,0].shape)

    print(target_1)
    print(target_2)
    print(target_3)

    figure = plt.figure(figsize=(60, 35))
    fontsize = 40

    ax_img_gc_1 = figure.add_axes([-0.12, 0.68, 0.5, 0.3], projection=wcs_hist_1)
    ax_img_cascade_1 = figure.add_axes([0.08, 0.68, 0.5, 0.3], projection=wcs_hist_1)
    ax_img_young_1 = figure.add_axes([0.27, 0.68, 0.5, 0.3], projection=wcs_hist_1)
    ax_img_rgb_1 = figure.add_axes([0.45, 0.68, 0.5, 0.3], projection=wcs_hist_1)
    ax_img_hst_rgb_1 = figure.add_axes([0.65, 0.68, 0.5, 0.3], projection=wcs_hist_1)

    ax_img_gc_2 = figure.add_axes([-0.12, 0.355, 0.5, 0.3], projection=wcs_hist_2)
    ax_img_cascade_2 = figure.add_axes([0.08, 0.355, 0.5, 0.3], projection=wcs_hist_2)
    ax_img_young_2 = figure.add_axes([0.27, 0.355, 0.5, 0.3], projection=wcs_hist_2)
    ax_img_rgb_2 = figure.add_axes([0.45, 0.355, 0.5, 0.3], projection=wcs_hist_2)
    ax_img_hst_rgb_2 = figure.add_axes([0.65, 0.355, 0.5, 0.3], projection=wcs_hist_2)

    if target_3 is not None:
        ax_img_gc_3 = figure.add_axes([-0.12, 0.03, 0.5, 0.3], projection=wcs_hist_3)
        ax_img_cascade_3 = figure.add_axes([0.08, 0.03, 0.5, 0.3], projection=wcs_hist_3)
        ax_img_young_3 = figure.add_axes([0.27, 0.03, 0.5, 0.3], projection=wcs_hist_3)
        ax_img_rgb_3 = figure.add_axes([0.45, 0.03, 0.5, 0.3], projection=wcs_hist_3)
        ax_img_hst_rgb_3 = figure.add_axes([0.65, 0.03, 0.5, 0.3], projection=wcs_hist_3)

    hdu_rgb_img_hst_1 = fits.open('/home/benutzer/data/PHANGS-HST/rgb_img/%s_BVI_RGB.fits' % (target_1))
    hdu_rgb_img_hst_data_r_1 = plotting_tools.reproject_image(data=hdu_rgb_img_hst_1[0].data[0],
                                                            wcs=WCS(hdu_rgb_img_hst_1[0].header, naxis=2),
                                                            new_wcs=wcs_hist_1, new_shape=young_image_1[:,:,0].shape)
    hdu_rgb_img_hst_data_g_1 = plotting_tools.reproject_image(data=hdu_rgb_img_hst_1[0].data[1],
                                                            wcs=WCS(hdu_rgb_img_hst_1[0].header, naxis=2),
                                                            new_wcs=wcs_hist_1, new_shape=young_image_1[:,:,0].shape)
    hdu_rgb_img_hst_data_b_1 = plotting_tools.reproject_image(data=hdu_rgb_img_hst_1[0].data[2],
                                                            wcs=WCS(hdu_rgb_img_hst_1[0].header, naxis=2),
                                                            new_wcs=wcs_hist_1, new_shape=young_image_1[:,:,0].shape)
    hdu_rgb_img_hst_data_1 = np.array([hdu_rgb_img_hst_data_r_1.T, hdu_rgb_img_hst_data_g_1.T, hdu_rgb_img_hst_data_b_1.T])

    ax_img_hst_rgb_1.imshow(hdu_rgb_img_hst_data_1.T)

    hdu_rgb_img_hst_2 = fits.open('/home/benutzer/data/PHANGS-HST/rgb_img/%s_BVI_RGB.fits' % (target_2))
    hdu_rgb_img_hst_data_r_2 = plotting_tools.reproject_image(data=hdu_rgb_img_hst_2[0].data[0],
                                                            wcs=WCS(hdu_rgb_img_hst_2[0].header, naxis=2),
                                                            new_wcs=wcs_hist_2, new_shape=young_image_2[:,:,0].shape)
    hdu_rgb_img_hst_data_g_2 = plotting_tools.reproject_image(data=hdu_rgb_img_hst_2[0].data[1],
                                                            wcs=WCS(hdu_rgb_img_hst_2[0].header, naxis=2),
                                                            new_wcs=wcs_hist_2, new_shape=young_image_2[:,:,0].shape)
    hdu_rgb_img_hst_data_b_2 = plotting_tools.reproject_image(data=hdu_rgb_img_hst_2[0].data[2],
                                                            wcs=WCS(hdu_rgb_img_hst_2[0].header, naxis=2),
                                                            new_wcs=wcs_hist_2, new_shape=young_image_2[:,:,0].shape)
    hdu_rgb_img_hst_data_2 = np.array([hdu_rgb_img_hst_data_r_2.T, hdu_rgb_img_hst_data_g_2.T, hdu_rgb_img_hst_data_b_2.T])

    ax_img_hst_rgb_2.imshow(hdu_rgb_img_hst_data_2.T)

    plot_panels(ax_img_r=ax_img_gc_1, ax_img_g=ax_img_cascade_1, ax_img_b=ax_img_young_1,
                ax_img_rgb=ax_img_rgb_1,
                img_r=gc_image_1, img_g=cascade_image_1, img_b=young_image_1,
                img_rgb=rgb_image_1,
                contour_data=alma_data_1, contour_color='gold', alpha=1.0, linewidths=2)

    plot_panels(ax_img_r=ax_img_gc_2, ax_img_g=ax_img_cascade_2, ax_img_b=ax_img_young_2,
                ax_img_rgb=ax_img_rgb_2,
                img_r=gc_image_2, img_g=cascade_image_2, img_b=young_image_2,
                img_rgb=rgb_image_2,
                contour_data=alma_data_2, contour_color='gold', alpha=1.0, linewidths=2)
    if target_3 is not None:
        hdu_rgb_img_hst_3 = fits.open('/home/benutzer/data/PHANGS-HST/rgb_img/%s_BVI_RGB.fits' % (target_3))
        hdu_rgb_img_hst_data_r_3 = plotting_tools.reproject_image(data=hdu_rgb_img_hst_3[0].data[0],
                                                                wcs=WCS(hdu_rgb_img_hst_3[0].header, naxis=2),
                                                                new_wcs=wcs_hist_3, new_shape=young_image_3[:,:,0].shape)
        hdu_rgb_img_hst_data_g_3 = plotting_tools.reproject_image(data=hdu_rgb_img_hst_3[0].data[1],
                                                                wcs=WCS(hdu_rgb_img_hst_3[0].header, naxis=2),
                                                                new_wcs=wcs_hist_3, new_shape=young_image_3[:,:,0].shape)
        hdu_rgb_img_hst_data_b_3 = plotting_tools.reproject_image(data=hdu_rgb_img_hst_3[0].data[2],
                                                                wcs=WCS(hdu_rgb_img_hst_3[0].header, naxis=2),
                                                                new_wcs=wcs_hist_3, new_shape=young_image_3[:,:,0].shape)
        hdu_rgb_img_hst_data_3 = np.array([hdu_rgb_img_hst_data_r_3.T, hdu_rgb_img_hst_data_g_3.T, hdu_rgb_img_hst_data_b_3.T])

        plot_panels(ax_img_r=ax_img_gc_3, ax_img_g=ax_img_cascade_3, ax_img_b=ax_img_young_3,
                    ax_img_rgb=ax_img_rgb_3,
                    img_r=gc_image_3, img_g=cascade_image_3, img_b=young_image_3,
                    img_rgb=rgb_image_3,
                    contour_data=alma_data_3, contour_color='gold', alpha=1.0, linewidths=2)
        ax_img_hst_rgb_3.imshow(hdu_rgb_img_hst_data_3.T)

    ax_img_gc_1.set_title('Old Globular Clusters (ML)', fontsize=fontsize)
    ax_img_cascade_1.set_title('Middle Aged Clusters (ML)', fontsize=fontsize)
    ax_img_young_1.set_title('Young Clusters (ML) + ALMA', fontsize=fontsize)
    ax_img_rgb_1.set_title('RGB (ML)', fontsize=fontsize)

    ax_img_gc_1.text(young_image_1[:,:,0].shape[0]*0.05, young_image_1[:,:,0].shape[1]*0.95, target_1.upper(),
                           horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
    ax_img_gc_2.text(young_image_2[:,:,0].shape[0]*0.05, young_image_2[:,:,0].shape[1]*0.95, target_2.upper(),
                           horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
    if target_3 is not None:
        ax_img_gc_3.text(young_image_3[:,:,0].shape[0]*0.05, young_image_3[:,:,0].shape[1]*0.95, target_3.upper(),
                               horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)



    plotting_tools.arr_axis_params(ax=ax_img_gc_1, ra_tick_label=True, dec_tick_label=True,
                                   ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                                   ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                   fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    plotting_tools.arr_axis_params(ax=ax_img_cascade_1, ra_tick_label=True, dec_tick_label=True,
                                   ra_axis_label=' ', dec_axis_label=' ',
                                   ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                   fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    plotting_tools.arr_axis_params(ax=ax_img_young_1, ra_tick_label=True, dec_tick_label=True,
                                   ra_axis_label=' ', dec_axis_label=' ',
                                   ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                   fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    plotting_tools.arr_axis_params(ax=ax_img_rgb_1, ra_tick_label=True, dec_tick_label=True,
                                   ra_axis_label=' ', dec_axis_label=' ',
                                   ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                   fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)


    if target_3 is None:

        plotting_tools.arr_axis_params(ax=ax_img_gc_2, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                                       ra_minpad=0.3, dec_minpad=-0.3, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
        plotting_tools.arr_axis_params(ax=ax_img_cascade_2, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
        plotting_tools.arr_axis_params(ax=ax_img_young_2, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
        plotting_tools.arr_axis_params(ax=ax_img_rgb_2, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    else:
        plotting_tools.arr_axis_params(ax=ax_img_gc_2, ra_tick_label=True, dec_tick_label=True,
                                   ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                                   ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                   fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
        plotting_tools.arr_axis_params(ax=ax_img_cascade_2, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
        plotting_tools.arr_axis_params(ax=ax_img_young_2, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
        plotting_tools.arr_axis_params(ax=ax_img_rgb_2, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)

        plotting_tools.arr_axis_params(ax=ax_img_gc_3, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                                       ra_minpad=0.3, dec_minpad=-0.3, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
        plotting_tools.arr_axis_params(ax=ax_img_cascade_3, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
        plotting_tools.arr_axis_params(ax=ax_img_young_3, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
        plotting_tools.arr_axis_params(ax=ax_img_rgb_3, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)


    plt.savefig('plot_output/spatial_dist_panel_%i.png' % index)
    plt.savefig('plot_output/spatial_dist_panel_%i.pdf' % index)


exit()








