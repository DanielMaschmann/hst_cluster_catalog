import numpy as np
from photometry_tools import helper_func as hf, plotting_tools
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve
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

def get_cc_pos(target, cutout):

    if target == 'ngc0628':
        mask_ml = (target_name_ml == 'ngc0628e') | (target_name_ml == 'ngc0628c')
    else:
        mask_ml = target_name_ml == target
    pos_ml = cutout.wcs.world_to_pixel(SkyCoord(ra=ra_ml[mask_ml]*u.deg, dec=dec_ml[mask_ml]*u.deg))
    x_ml, y_ml = pos_ml[1], pos_ml[0]
    mask_gc_ml = in_hull_gc_ml[mask_ml] * (age_ml[mask_ml] > 10)
    mask_cascade_ml = in_hull_cascade_ml[mask_ml] * (age_ml[mask_ml] > 10)
    mask_young_ml = in_hull_young_ml[mask_ml] + (age_ml[mask_ml] < 10)

    return {'x_ml': x_ml, 'y_ml': y_ml, 'mask_gc_ml': mask_gc_ml, 'mask_cascade_ml': mask_cascade_ml,
            'mask_young_ml': mask_young_ml}


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

kernal_size_prop = 0.02

for index in range(13):

    index = 12
    print(index)
    target_1 = list(cut_out_size_dict.keys())[index * 3]
    tab_1 = Simbad.query_object(target_1)
    central_coords_1 = SkyCoord('%s %s' % (tab_1['RA'].value[0], tab_1['DEC'].value[0]), unit=(u.hourangle, u.deg))
    target_2 = list(cut_out_size_dict.keys())[index * 3 + 1]
    tab_2 = Simbad.query_object(target_2)
    central_coords_2 = SkyCoord('%s %s' % (tab_2['RA'].value[0], tab_2['DEC'].value[0]), unit=(u.hourangle, u.deg))
    if index == 12:
        target_3 = None
        central_coords_3 = None
    else:
        target_3 = list(cut_out_size_dict.keys())[index * 3 + 2]
        tab_3 = Simbad.query_object(target_3)
        central_coords_3 = SkyCoord('%s %s' % (tab_3['RA'].value[0], tab_3['DEC'].value[0]), unit=(u.hourangle, u.deg))

    print(target_1)
    print(target_2)
    print(target_3)

    cutout_size_1 = cut_out_size_dict[target_1]
    alma_hdu_1 = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' % (target_1, target_1))
    cutout_alma_1 = hf.get_img_cutout(img=alma_hdu_1[0].data, wcs=WCS(alma_hdu_1[0].header),
                                      coord=central_coords_1, cutout_size=cutout_size_1)
    cutout_size_2 = cut_out_size_dict[target_2]
    alma_hdu_2 = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' % (target_2, target_2))
    cutout_alma_2 = hf.get_img_cutout(img=alma_hdu_2[0].data, wcs=WCS(alma_hdu_2[0].header),
                                  coord=central_coords_2, cutout_size=cutout_size_2)
    if target_3 is None:
        cutout_alma_3 = None
    else:
        cutout_size_3 = cut_out_size_dict[target_3]
        alma_hdu_3 = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' % (target_3, target_3))
        cutout_alma_3 = hf.get_img_cutout(img=alma_hdu_3[0].data, wcs=WCS(alma_hdu_3[0].header),
                                          coord=central_coords_3, cutout_size=cutout_size_3)

    # create cluster images
    pos_dict_1 = get_cc_pos(target=target_1, cutout=cutout_alma_1)
    pos_dict_2 = get_cc_pos(target=target_2, cutout=cutout_alma_2)
    if target_3 is None:
        pos_dict_3 = None
    else:
        pos_dict_3 = get_cc_pos(target=target_3, cutout=cutout_alma_3)

    # make images
    shape_1 = cutout_alma_1.data.shape
    shape_2 = cutout_alma_2.data.shape
    gc_image_1_ml, cascade_image_1_ml, young_image_1_ml, rgb_image_1_ml = (
        get_rgb_images(kernal_size_prop=kernal_size_prop, shape=shape_1,
                       x=pos_dict_1['x_ml'], y=pos_dict_1['y_ml'],
                       mask_r=pos_dict_1['mask_gc_ml'], mask_g=pos_dict_1['mask_cascade_ml'],
                       mask_b=pos_dict_1['mask_young_ml']))
    gc_image_2_ml, cascade_image_2_ml, young_image_2_ml, rgb_image_2_ml = (
        get_rgb_images(kernal_size_prop=kernal_size_prop, shape=shape_2,
                       x=pos_dict_2['x_ml'], y=pos_dict_2['y_ml'],
                       mask_r=pos_dict_2['mask_gc_ml'], mask_g=pos_dict_2['mask_cascade_ml'],
                       mask_b=pos_dict_2['mask_young_ml']))
    if target_3 is None:
        gc_image_3_ml, cascade_image_3_ml, young_image_3_ml, rgb_image_3_ml = None, None, None, None
    else:
        shape_3 = cutout_alma_3.data.shape
        gc_image_3_ml, cascade_image_3_ml, young_image_3_ml, rgb_image_3_ml = (
            get_rgb_images(kernal_size_prop=kernal_size_prop, shape=shape_3,
                           x=pos_dict_3['x_ml'], y=pos_dict_3['y_ml'],
                           mask_r=pos_dict_3['mask_gc_ml'], mask_g=pos_dict_3['mask_cascade_ml'],
                           mask_b=pos_dict_3['mask_young_ml']))


    figure = plt.figure(figsize=(48, 35))
    fontsize = 40

    ax_img_gc_1 = figure.add_axes([-0.11, 0.68, 0.5, 0.3], projection=cutout_alma_1.wcs)
    ax_img_cascade_1 = figure.add_axes([0.135, 0.68, 0.5, 0.3], projection=cutout_alma_1.wcs)
    ax_img_young_1 = figure.add_axes([0.38, 0.68, 0.5, 0.3], projection=cutout_alma_1.wcs)
    ax_img_rgb_1 = figure.add_axes([0.625, 0.68, 0.5, 0.3], projection=cutout_alma_1.wcs)

    ax_img_gc_2 = figure.add_axes([-0.11, 0.355, 0.5, 0.3], projection=cutout_alma_2.wcs)
    ax_img_cascade_2 = figure.add_axes([0.135, 0.355, 0.5, 0.3], projection=cutout_alma_2.wcs)
    ax_img_young_2 = figure.add_axes([0.38, 0.355, 0.5, 0.3], projection=cutout_alma_2.wcs)
    ax_img_rgb_2 = figure.add_axes([0.625, 0.355, 0.5, 0.3], projection=cutout_alma_2.wcs)

    if target_3 is not None:
        ax_img_gc_3 = figure.add_axes([-0.11, 0.03, 0.5, 0.3], projection=cutout_alma_3.wcs)
        ax_img_cascade_3 = figure.add_axes([0.135, 0.03, 0.5, 0.3], projection=cutout_alma_3.wcs)
        ax_img_young_3 = figure.add_axes([0.38, 0.03, 0.5, 0.3], projection=cutout_alma_3.wcs)
        ax_img_rgb_3 = figure.add_axes([0.625, 0.03, 0.5, 0.3], projection=cutout_alma_3.wcs)


    plot_panels(ax_img_r=ax_img_gc_1, ax_img_g=ax_img_cascade_1, ax_img_b=ax_img_young_1,
                ax_img_rgb=ax_img_rgb_1,
                img_r=gc_image_1_ml, img_g=cascade_image_1_ml, img_b=young_image_1_ml,
                img_rgb=rgb_image_1_ml,
                contour_data=cutout_alma_1.data, contour_color='gold', alpha=1.0, linewidths=2)

    plot_panels(ax_img_r=ax_img_gc_2, ax_img_g=ax_img_cascade_2, ax_img_b=ax_img_young_2,
                ax_img_rgb=ax_img_rgb_2,
                img_r=gc_image_2_ml, img_g=cascade_image_2_ml, img_b=young_image_2_ml,
                img_rgb=rgb_image_2_ml,
                contour_data=cutout_alma_2.data, contour_color='gold', alpha=1.0, linewidths=2)
    if target_3 is not None:
        plot_panels(ax_img_r=ax_img_gc_3, ax_img_g=ax_img_cascade_3, ax_img_b=ax_img_young_3,
                    ax_img_rgb=ax_img_rgb_3,
                    img_r=gc_image_3_ml, img_g=cascade_image_3_ml, img_b=young_image_3_ml,
                    img_rgb=rgb_image_3_ml,
                    contour_data=cutout_alma_3.data, contour_color='gold', alpha=1.0, linewidths=2)

    ax_img_gc_1.set_title('Old Globular Clusters (ML)', fontsize=fontsize)
    ax_img_cascade_1.set_title('Middle Aged Clusters (ML)', fontsize=fontsize)
    ax_img_young_1.set_title('Young Clusters (ML) + ALMA', fontsize=fontsize)
    ax_img_rgb_1.set_title('RGB (ML)', fontsize=fontsize)

    ax_img_gc_1.text(shape_1[0]*0.05, shape_1[1]*0.95, target_1.upper(),
                           horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
    ax_img_gc_2.text(shape_2[0]*0.05, shape_2[1]*0.95, target_2.upper(),
                           horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
    if target_3 is not None:
        ax_img_gc_3.text(shape_3[0]*0.05, shape_3[1]*0.95, target_3.upper(),
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
