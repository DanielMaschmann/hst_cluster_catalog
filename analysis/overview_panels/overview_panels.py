import numpy as np
from photometry_tools import helper_func as hf, plotting_tools
from photometry_tools.data_access import CatalogAccess
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from scipy.stats import gaussian_kde
from astropy.convolution import convolve
import multicolorfits as mcf

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.spatial import ConvexHull
from astroquery.simbad import Simbad


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

model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')


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


def contours(ax, x, y, levels=None, axis_offse=(-0.2, 0.1, -0.55, 0.6), linewidth=4):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]

    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    x = x[good_values]
    y = y[good_values]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min()+axis_offse[0]:x.max()+axis_offse[1]:x.size**0.5*1j,
             y.min()+axis_offse[2]:y.max()+axis_offse[3]:y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)
    # ax[0].scatter(xi.flatten(), yi.flatten(), c=zi)
    cs = ax.contour(xi, yi, zi, levels=levels,
                    colors='k',
                    linewidths=(linewidth,),
                    origin='lower')


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


    bkg_estimator = MedianBackground()
    sigma_clip = SigmaClip(sigma=3.0)
    contour_data[np.isnan(contour_data)] = 0
    bkg = Background2D(contour_data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)


    # print(bkg.background)
    # print(contour_data - bkg.background)
    contour_data -= bkg.background
    # exit()
    #
    mean, median, std = sigma_clipped_stats(contour_data, sigma=3.0)
    levels_std = np.array(np.arange(start=0.3, stop=5, step=1), dtype=float)
    levels = np.ones(len(levels_std)) * mean
    levels += levels_std

    # from astropy.stats import SigmaClip
    # from photutils.background import Background2D, MedianBackground
    # bkg_estimator = MedianBackground()
    # sigma_clip = SigmaClip(sigma=3.0)
    # print(contour_data.shape)
    # bkg = Background2D(contour_data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    # contour_data = contour_data - bkg.background
    # mean, median, std = sigma_clipped_stats(contour_data, sigma=3.0)
    # levels_std = np.array(np.arange(start=0.3, stop=5, step=1), dtype=float)
    # levels = np.ones(len(levels_std)) * mean
    # levels += levels_std
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

catalog_access = CatalogAccess()

n_bins = 250
add_side_pixel_frac = 3/100
kernel_size_prop = 0.02

color_c1 = 'darkorange'
color_c2 = 'forestgreen'
color_c3 = 'royalblue'
vi_int = 0.6
ub_int = -2.0
nuvb_int = -3.0
av_value = 1

x_lim_vi = (-1.0, 2.3)
y_lim_ub = (1.25, -2.8)
y_lim_nuvb = (3.2, -4.3)

mask_cl1_hum = clcl_color_hum == 1
mask_cl2_hum = clcl_color_hum == 2
mask_cl12_hum = (clcl_color_hum == 1) | (clcl_color_hum == 2)
mask_cl3_hum = clcl_color_hum == 3

mask_cl1_ml = clcl_color_ml == 1
mask_cl2_ml = clcl_color_ml == 2
mask_cl12_ml = (clcl_color_ml == 1) | (clcl_color_ml == 2)
mask_cl3_ml = clcl_color_ml == 3

good_colors_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                   (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                   detect_v_hum & detect_i_hum & detect_u_hum & detect_b_hum)

good_colors_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                   (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                   detect_v_ml & detect_i_ml & detect_u_ml & detect_b_ml)


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


mean_dist_gc = np.zeros(len(galaxy_name_list))
median_dist_gc = np.zeros(len(galaxy_name_list))
std_dist_gc = np.zeros(len(galaxy_name_list))

mean_dist_cascade = np.zeros(len(galaxy_name_list))
median_dist_cascade = np.zeros(len(galaxy_name_list))
std_dist_cascade = np.zeros(len(galaxy_name_list))

mean_dist_young = np.zeros(len(galaxy_name_list))
median_dist_young = np.zeros(len(galaxy_name_list))
std_dist_young = np.zeros(len(galaxy_name_list))

mean_dist_random = np.zeros(len(galaxy_name_list))
median_dist_random = np.zeros(len(galaxy_name_list))
std_dist_random = np.zeros(len(galaxy_name_list))


for index_galaxy_name, galaxy_name in enumerate(galaxy_name_list):
    # galaxy_name = 'ngc6744'


    if galaxy_name == 'ngc0628':
        target_mask_ml = (target_name_ml == 'ngc0628e') | (target_name_ml == 'ngc0628c')
        target_mask_hum = (target_name_hum == 'ngc0628e') | (target_name_hum == 'ngc0628c')
    else:
        target_mask_ml = target_name_ml == galaxy_name
        target_mask_hum = target_name_hum == galaxy_name

    gc_image, cascade_image, young_image, rgb_image, wcs_hist = create_hist(galaxy_name=galaxy_name,
                                                                            n_bins=n_bins,
                                                                            add_side_pixel_frac=add_side_pixel_frac,
                                                                            kernel_size_prop=kernel_size_prop)
    # gte alma observation data
    alma_hdu = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' %
                         (galaxy_name, galaxy_name))
    alma_wcs = WCS(alma_hdu[0].header)
    contours_alma_coverage = np.invert(np.isnan(alma_hdu[0].data))

    alma_data_reprojected = plotting_tools.reproject_image(data=alma_hdu[0].data, wcs=WCS(alma_hdu[0].header),
                                                           new_wcs=wcs_hist, new_shape=young_image[:,:,0].shape)
    alma_borders_reprojected = np.invert(np.isnan(alma_data_reprojected))

    # get the coordinates of molecular clouds
    cloud_cat_hdu = fits.open('/home/benutzer/data/PHANGS_products/cloud_catalogs/v4p0_ST1p6/v4p0_gmccats/native/%s_12m+7m+tp_co21_native_props.fits' % galaxy_name)
    print(cloud_cat_hdu[1].data.names)
    exit()
    x_cloud = cloud_cat_hdu[1].data['XCTR_PIX']
    y_cloud = cloud_cat_hdu[1].data['YCTR_PIX']
    ra_cloud = cloud_cat_hdu[1].data['XMAX_DEG']
    dec_cloud = cloud_cat_hdu[1].data['YMAX_DEG']
    # create sky coordinates for all the clouds
    coords_cloud = SkyCoord(ra=ra_cloud*u.deg, dec=dec_cloud*u.deg)

    # create sky coords for all clusters
    coords_cluster = SkyCoord(ra=ra_ml[target_mask_ml]*u.deg, dec=dec_ml[target_mask_ml]*u.deg)

    # calculate a mask of clujsters inside the alma observation
    coords_cluster_on_alma_pos = alma_wcs.world_to_pixel(coords_cluster)
    x_coords_on_alma = coords_cluster_on_alma_pos[0]
    y_coords_on_alma = coords_cluster_on_alma_pos[1]
    mask_outside_alma = ((x_coords_on_alma < 0) | (y_coords_on_alma < 0) |
                         (x_coords_on_alma > alma_hdu[0].data.shape[1]) |
                         (y_coords_on_alma > alma_hdu[0].data.shape[0]))

    mask_outside_alma_coverage = mask_outside_alma.copy()
    mask_outside_alma_coverage[np.invert(mask_outside_alma)] = (
        np.invert(contours_alma_coverage[np.array(y_coords_on_alma[np.invert(mask_outside_alma)], dtype=int),
                                         np.array(x_coords_on_alma[np.invert(mask_outside_alma)], dtype=int)]))
    mask_alma_coverage = np.invert(mask_outside_alma_coverage)


    # get random corrdinates
    x_random_on_alma = np.random.uniform(low=0, high=alma_hdu[0].data.shape[1], size=sum(target_mask_ml))
    y_random_on_alma = np.random.uniform(low=0, high=alma_hdu[0].data.shape[0], size=sum(target_mask_ml))
    mask_random_on_alma = (
        np.invert(contours_alma_coverage[np.array(y_random_on_alma, dtype=int), np.array(x_random_on_alma, dtype=int)]))
    mask_random_alma_coverage = np.invert(mask_random_on_alma)

    # # test code to inspect positions
    # figure = plt.figure(figsize=(15, 15))
    # fontsize = 23
    # ax_alma = figure.add_axes([0.05, 0.05, 0.9, 0.9], projection=alma_wcs)
    # ax_alma.imshow(alma_hdu[0].data)
    # # ax_alma.scatter(x_coords_on_alma, y_coords_on_alma)#, c=min_dist2arm_array)
    # # ax_alma.scatter(x_coords_on_alma[mask_alma_coverage], y_coords_on_alma[mask_alma_coverage], c='red')
    # ax_alma.contour(contours_alma_coverage, levels=0, colors='k', linewidths=4)
    #
    # ax_alma.scatter(x_random_on_alma, y_random_on_alma)
    # ax_alma.scatter(x_random_on_alma[mask_random_alma_coverage], y_random_on_alma[mask_random_alma_coverage])
    #
    # plt.show()

    simbad_table = Simbad.query_object(galaxy_name)
    central_coordinates = SkyCoord('%s %s' % (simbad_table['RA'].value[0],  simbad_table['DEC'].value[0]),  unit=(u.hourangle, u.deg))
    coords_alma_center = alma_wcs.world_to_pixel(central_coordinates)
    x_center_on_alma = coords_alma_center[0]
    y_center_on_alma = coords_alma_center[1]

    min_dist2cloud_array = np.zeros(len(x_coords_on_alma))
    min_dist2cloud_random_array = np.zeros(len(x_coords_on_alma))
    dist2center_array = np.zeros(len(x_coords_on_alma))

    for index in range(len(min_dist2cloud_array)):
        # now get the distance to the arms
        dist2cloud = np.sqrt((x_cloud - x_coords_on_alma[index]) ** 2 + (y_cloud - y_coords_on_alma[index]) ** 2)
        min_dist2cloud_arcsec = hf.transform_pix2world_scale(pixel_length=min(dist2cloud), wcs=alma_wcs)
        min_dist2cloud_pc = hf.arcsec2kpc(dist_arcsec=min_dist2cloud_arcsec, dist=catalog_access.dist_dict[galaxy_name]['dist']) * 1e3
        min_dist2cloud_array[index] = min_dist2cloud_pc

        dist2cloud_random = np.sqrt((x_cloud - x_random_on_alma[index]) ** 2 + (y_cloud - y_random_on_alma[index]) ** 2)
        min_dist2cloud_random_arcsec = hf.transform_pix2world_scale(pixel_length=min(dist2cloud_random), wcs=alma_wcs)
        min_dist2cloud_random_pc = hf.arcsec2kpc(dist_arcsec=min_dist2cloud_random_arcsec, dist=catalog_access.dist_dict[galaxy_name]['dist']) * 1e3
        min_dist2cloud_random_array[index] = min_dist2cloud_random_pc

        dist2center = np.sqrt((x_center_on_alma - x_coords_on_alma[index]) ** 2 + (y_center_on_alma - y_coords_on_alma[index]) ** 2)
        dist2center_arcsec = hf.transform_pix2world_scale(pixel_length=dist2center, wcs=alma_wcs)
        dist2center_pc = hf.arcsec2kpc(dist_arcsec=dist2center_arcsec, dist=catalog_access.dist_dict[galaxy_name]['dist']) * 1e3
        dist2center_array[index] = dist2center_pc

    # figure = plt.figure(figsize=(15, 15))
    # fontsize = 23
    # ax_alma = figure.add_axes([0.05, 0.05, 0.9, 0.9], projection=alma_wcs)
    # ax_alma.imshow(alma_hdu[0].data, cmap='Greys')
    # ax_alma.scatter(x_coords_on_alma, y_coords_on_alma, c=min_dist2cloud_array, vmin=0, vmax=0.5)
    # ax_alma.scatter(x_coords_on_alma[np.invert(mask_alma_coverage)], y_coords_on_alma[np.invert(mask_alma_coverage)], c='red')
    # plt.show()



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



    figure = plt.figure(figsize=(70, 50))
    fontsize = 40

    ax_ubvi_12 = figure.add_axes([0.03, 0.695, 0.23, 0.3])
    ax_ubvi_3 = figure.add_axes([0.265, 0.695, 0.23, 0.3])
    ax_age_ebv = figure.add_axes([0.525, 0.695, 0.23, 0.3])
    ax_img_hst_rgb = figure.add_axes([0.64, 0.695, 0.5, 0.3], projection=wcs_hist)

    ax_img_gc = figure.add_axes([-0.10, 0.35, 0.5, 0.3], projection=wcs_hist)
    ax_img_cascade = figure.add_axes([0.15, 0.35, 0.5, 0.3], projection=wcs_hist)
    ax_img_young = figure.add_axes([0.40, 0.35, 0.5, 0.3], projection=wcs_hist)
    ax_img_rgb = figure.add_axes([0.64, 0.35, 0.5, 0.3], projection=wcs_hist)

    ax_sepa_gc = figure.add_axes([0.03, 0.03, 0.23, 0.3])
    ax_sepa_cascade = figure.add_axes([0.27, 0.03, 0.23, 0.3])
    ax_sepa_young = figure.add_axes([0.51, 0.03, 0.23, 0.3])
    ax_sepa_random = figure.add_axes([0.76, 0.03, 0.23, 0.3])


    contours(ax=ax_ubvi_12, x=color_vi_ml[target_mask_ml*mask_cl12_ml*good_colors_ml],
             y=color_ub_ml[target_mask_ml*mask_cl12_ml*good_colors_ml])
    ax_ubvi_12.plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=4, zorder=10)

    ax_ubvi_12.scatter(color_vi_hum[target_mask_hum*mask_cl1_hum*good_colors_hum],
                       color_ub_hum[target_mask_hum*mask_cl1_hum*good_colors_hum], c=color_c1, s=140)
    ax_ubvi_12.scatter(color_vi_hum[target_mask_hum*mask_cl2_hum*good_colors_hum],
                       color_ub_hum[target_mask_hum*mask_cl2_hum*good_colors_hum], c=color_c2, s=140)
    hf.plot_reddening_vect(ax=ax_ubvi_12, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=3, line_color='r', text=True, x_text_offset=0.0, y_text_offset=-0.05, fontsize=fontsize)

    contours(ax=ax_ubvi_3, x=color_vi_ml[target_mask_ml*mask_cl3_ml*good_colors_ml],
             y=color_ub_ml[target_mask_ml*mask_cl3_ml*good_colors_ml])
    ax_ubvi_3.plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=4, zorder=10)

    ax_ubvi_3.scatter(color_vi_hum[target_mask_hum*mask_cl3_hum*good_colors_hum],
                       color_ub_hum[target_mask_hum*mask_cl3_hum*good_colors_hum], c=color_c3, s=140)
    hf.plot_reddening_vect(ax=ax_ubvi_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=3, line_color='r', text=True, x_text_offset=0.0, y_text_offset=-0.05, fontsize=fontsize)
    if 'F438W' in catalog_access.hst_targets[galaxy_name]['wfc3_uvis_observed_bands']:
        ax_ubvi_12.set_ylabel('U (F336W) - B (F438W)', fontsize=fontsize)
    else:
        ax_ubvi_12.set_ylabel('U (F336W) - B (F435W)', fontsize=fontsize)

    ax_ubvi_12.scatter([], [], c=color_c1, s=140, label='Class 1 (Hum)')
    ax_ubvi_12.scatter([], [], c=color_c2, s=140, label='Class 2 (Hum)')
    ax_ubvi_12.plot([], [], color='k', linewidth=4, label='Class 1|2 (ML)')
    ax_ubvi_12.legend(frameon=False, fontsize=fontsize)

    ax2 = ax_ubvi_12.twinx()
    ax2.plot(np.NaN, np.NaN, color='tab:red', linewidth=4, label=r'BC03, Z$_{\odot}$')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc=3, frameon=False, fontsize=fontsize)

    ax_ubvi_3.scatter([], [], c=color_c3, s=140, label='compact association (Hum)')
    ax_ubvi_3.plot([], [], color='k', linewidth=4, label='Compact association (ML)')
    ax_ubvi_3.legend(frameon=False, fontsize=fontsize)

    ax_ubvi_12.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
    ax_ubvi_3.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

    ax_ubvi_12.set_ylim(y_lim_ub)
    ax_ubvi_12.set_xlim(x_lim_vi)
    ax_ubvi_3.set_ylim(y_lim_ub)
    ax_ubvi_3.set_xlim(x_lim_vi)
    ax_ubvi_3.set_yticks([])
    ax_ubvi_12.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_ubvi_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

    # random dots
    random_x_hum = np.random.uniform(low=-0.1, high=0.1, size=len(age_hum[target_mask_hum]))
    random_y_hum = np.random.uniform(low=-0.05, high=0.05, size=len(age_hum[target_mask_hum]))
    ax_age_ebv.scatter((np.log10(age_hum[target_mask_hum]) + random_x_hum + 6)[mask_cl3_hum[target_mask_hum]],
                       (ebv_hum[target_mask_hum] + random_y_hum)[mask_cl3_hum[target_mask_hum]], c=color_c3, s=140, label='Compact associations (hum)')
    ax_age_ebv.scatter((np.log10(age_hum[target_mask_hum]) + random_x_hum + 6)[mask_cl2_hum[target_mask_hum]],
                       (ebv_hum[target_mask_hum] + random_y_hum)[mask_cl2_hum[target_mask_hum]], c=color_c2, s=140, label='Class 2 (hum)')
    ax_age_ebv.scatter((np.log10(age_hum[target_mask_hum]) + random_x_hum + 6)[mask_cl1_hum[target_mask_hum]],
                       (ebv_hum[target_mask_hum] + random_y_hum)[mask_cl1_hum[target_mask_hum]], c=color_c1, s=140, label='Class 1 (hum)')
    ax_age_ebv.set_xlim(5.7, 10.3)
    ax_age_ebv.set_ylim(-0.1, 2.1)
    ax_age_ebv.legend(frameon=False, fontsize=fontsize)
    ax_age_ebv.set_xlabel('log(Age/yr) + random.uniform(-0.1, 0.1)', fontsize=fontsize)
    ax_age_ebv.set_ylabel('E(B-V) + random.uniform(-0.05, 0.05)', fontsize=fontsize)
    ax_age_ebv.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

    grey_r = mcf.greyRGBize_image(rgb_img_hst_data.T[:,:,0], rescalefn='asinh', scaletype='perc', min_max=[0.3, 99.5],
                                      gamma=17.5, checkscale=False)
    grey_g = mcf.greyRGBize_image(rgb_img_hst_data.T[:,:,1], rescalefn='asinh', scaletype='perc', min_max=[0.3, 99.5],
                                      gamma=17.5, checkscale=False)
    grey_b = mcf.greyRGBize_image(rgb_img_hst_data.T[:,:,2], rescalefn='asinh', scaletype='perc', min_max=[0.3, 99.5],
                                      gamma=17.5, checkscale=False)
    r = mcf.colorize_image(grey_r, '#FF4433', colorintype='hex', gammacorr_color=17.5)
    g = mcf.colorize_image(grey_g, '#0FFF50', colorintype='hex', gammacorr_color=17.5)
    b = mcf.colorize_image(grey_b, '#0096FF', colorintype='hex', gammacorr_color=17.5)
    hst_rgb_image = mcf.combine_multicolor([r, g, b], gamma=17.5, inverse=False)
    ax_img_hst_rgb.imshow(hst_rgb_image)
    # ax_img_hst_rgb.imshow(rgb_img_hst_data.T)
    ax_img_hst_rgb.text(rgb_img_hst_data[0, :, :].shape[0]*0.05, rgb_img_hst_data[0, :, :].shape[1]*0.95, galaxy_name.upper(),
                        color='white', horizontalalignment='left', verticalalignment='top', fontsize=fontsize + 5)
    plot_panels(ax_img_r=ax_img_gc, ax_img_g=ax_img_cascade, ax_img_b=ax_img_young,
                    ax_img_rgb=ax_img_rgb,
                    img_r=gc_image, img_g=cascade_image, img_b=young_image,
                    img_rgb=rgb_image,
                    contour_data=alma_data_reprojected, contour_color='gold', alpha=1.0, linewidths=2)

    ax_img_gc.set_title('Old Globular Clusters (ML)', fontsize=fontsize)
    ax_img_cascade.set_title('Middle Aged Clusters (ML)', fontsize=fontsize)
    ax_img_young.set_title('Young Clusters (ML) + ALMA', fontsize=fontsize)
    ax_img_rgb.set_title('RGB (ML)', fontsize=fontsize)

    plotting_tools.arr_axis_params(ax=ax_img_hst_rgb, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)

    plotting_tools.arr_axis_params(ax=ax_img_gc, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)


    plotting_tools.arr_axis_params(ax=ax_img_cascade, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    plotting_tools.arr_axis_params(ax=ax_img_young, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
    plotting_tools.arr_axis_params(ax=ax_img_rgb, ra_tick_label=True, dec_tick_label=True,
                                       ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                                       ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                                       fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)





    ax_img_young.contour(alma_borders_reprojected, levels=0, colors='k', linewidths=4)

    mask_gc_ml = (in_hull_gc_ml * (age_ml >= 10))[target_mask_ml]
    mask_cascade_ml = (in_hull_cascade_ml * (age_ml >= 10))[target_mask_ml]
    mask_young_ml = (in_hull_young_ml + (age_ml < 10))[target_mask_ml]
    mask_good_dist = min_dist2cloud_array < 1e4
    mask_good_dist_ramndom = min_dist2cloud_random_array < 1e4

    mean_dist_gc[index_galaxy_name] = np.nanmean(min_dist2cloud_array[mask_gc_ml * mask_alma_coverage * mask_good_dist])
    median_dist_gc[index_galaxy_name] = np.nanmedian(min_dist2cloud_array[mask_gc_ml * mask_alma_coverage * mask_good_dist])
    std_dist_gc[index_galaxy_name] = np.nanstd(min_dist2cloud_array[mask_gc_ml * mask_alma_coverage * mask_good_dist])

    mean_dist_cascade[index_galaxy_name] = np.nanmean(min_dist2cloud_array[mask_cascade_ml * mask_alma_coverage * mask_good_dist])
    median_dist_cascade[index_galaxy_name] = np.nanmedian(min_dist2cloud_array[mask_cascade_ml * mask_alma_coverage * mask_good_dist])
    std_dist_cascade[index_galaxy_name] = np.nanstd(min_dist2cloud_array[mask_cascade_ml * mask_alma_coverage * mask_good_dist])

    mean_dist_young[index_galaxy_name] = np.nanmean(min_dist2cloud_array[mask_young_ml * mask_alma_coverage * mask_good_dist])
    median_dist_young[index_galaxy_name] = np.nanmedian(min_dist2cloud_array[mask_young_ml * mask_alma_coverage * mask_good_dist])
    std_dist_young[index_galaxy_name] = np.nanstd(min_dist2cloud_array[mask_young_ml * mask_alma_coverage * mask_good_dist])

    mean_dist_random[index_galaxy_name] = np.nanmean(min_dist2cloud_random_array[mask_random_alma_coverage * mask_good_dist_ramndom])
    median_dist_random[index_galaxy_name] = np.nanmedian(min_dist2cloud_random_array[mask_random_alma_coverage * mask_good_dist_ramndom])
    std_dist_random[index_galaxy_name] = np.nanstd(min_dist2cloud_random_array[mask_random_alma_coverage * mask_good_dist_ramndom])



    dist_bins = np.logspace(0.5, 4, 20)
    center_of_dist_bins = (dist_bins[:-1] + dist_bins[1:]) / 2
    hist_sepa_gc = np.histogram(min_dist2cloud_array[mask_gc_ml * mask_alma_coverage], bins=dist_bins)[0]
    hist_sepa_cascade = np.histogram(min_dist2cloud_array[mask_cascade_ml * mask_alma_coverage], bins=dist_bins)[0]
    hist_sepa_young = np.histogram(min_dist2cloud_array[mask_young_ml * mask_alma_coverage], bins=dist_bins)[0]
    hist_sepa_random = np.histogram(min_dist2cloud_random_array[mask_random_alma_coverage], bins=dist_bins)[0]

    ax_sepa_gc.step(center_of_dist_bins, hist_sepa_gc, where='mid', linewidth=4, color='red', label='Old Globular Clusters (ML)')
    ax_sepa_cascade.step(center_of_dist_bins, hist_sepa_cascade, where='mid', linewidth=4, color='green', label='Middle Aged Clusters (ML)')
    ax_sepa_young.step(center_of_dist_bins, hist_sepa_young, where='mid', linewidth=4, color='blue', label='Young Clusters (ML)')
    ax_sepa_random.step(center_of_dist_bins, hist_sepa_random, where='mid', linewidth=4, color='k', label='Random')

    ax_sepa_gc.plot([median_dist_gc[index_galaxy_name], median_dist_gc[index_galaxy_name]], [0, np.max(hist_sepa_gc)], linewidth=4, linestyle='--', color='k')
    ax_sepa_cascade.plot([median_dist_cascade[index_galaxy_name], median_dist_cascade[index_galaxy_name]], [0, np.max(hist_sepa_cascade)], linewidth=4, linestyle='--', color='k')
    ax_sepa_young.plot([median_dist_young[index_galaxy_name], median_dist_young[index_galaxy_name]], [0, np.max(hist_sepa_young)], linewidth=4, linestyle='--', color='k')
    ax_sepa_random.plot([median_dist_random[index_galaxy_name], median_dist_random[index_galaxy_name]], [0, np.max(hist_sepa_random)], linewidth=4, linestyle='--', color='k')

    ax_sepa_gc.set_xscale('log')
    ax_sepa_cascade.set_xscale('log')
    ax_sepa_young.set_xscale('log')
    ax_sepa_random.set_xscale('log')

    ax_sepa_gc.legend(frameon=False, loc=2, fontsize=fontsize)
    ax_sepa_cascade.legend(frameon=False, loc=2, fontsize=fontsize)
    ax_sepa_young.legend(frameon=False, loc=2, fontsize=fontsize)
    ax_sepa_random.legend(frameon=False, loc=2, fontsize=fontsize)
    ax_sepa_gc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_sepa_cascade.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_sepa_young.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_sepa_random.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

    ax_sepa_gc.set_ylabel('# Counts', fontsize=fontsize)

    ax_sepa_gc.set_xlabel('Dist to GMC [pc]', fontsize=fontsize)
    ax_sepa_cascade.set_xlabel('Dist to GMC [pc]', fontsize=fontsize)
    ax_sepa_young.set_xlabel('Dist to GMC [pc]', fontsize=fontsize)
    ax_sepa_random.set_xlabel('Dist to GMC [pc]', fontsize=fontsize)


    plt.savefig('plot_output/overview_panel_%s.png' % galaxy_name)
    plt.cla()
    plt.close()
    plt.clf()


np.save('data_output/mean_dist_gc.npy', mean_dist_gc)
np.save('data_output/median_dist_gc.npy', median_dist_gc)
np.save('data_output/std_dist_gc.npy', std_dist_gc)
np.save('data_output/mean_dist_cascade.npy', mean_dist_cascade)
np.save('data_output/median_dist_cascade.npy', median_dist_cascade)
np.save('data_output/std_dist_cascade.npy', std_dist_cascade)
np.save('data_output/mean_dist_young.npy', mean_dist_young)
np.save('data_output/median_dist_young.npy', median_dist_young)
np.save('data_output/std_dist_young.npy', std_dist_young)
np.save('data_output/mean_dist_random.npy', mean_dist_random)
np.save('data_output/median_dist_random.npy', median_dist_random)
np.save('data_output/std_dist_random.npy', std_dist_random)


exit()













