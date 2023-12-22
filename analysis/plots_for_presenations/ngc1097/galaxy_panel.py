import numpy as np
from photometry_tools import helper_func as hf, plotting_tools
from photometry_tools.data_access import CatalogAccess
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
import multicolorfits as mcf

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.spatial import ConvexHull
from matplotlib import patheffects


# get color-color data
color_vi_hum = np.load('../../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../../color_color/data_output/color_ub_hum.npy')
detect_u_hum = np.load('../../color_color/data_output/detect_u_hum.npy')
detect_b_hum = np.load('../../color_color/data_output/detect_b_hum.npy')
detect_v_hum = np.load('../../color_color/data_output/detect_v_hum.npy')
detect_i_hum = np.load('../../color_color/data_output/detect_i_hum.npy')
x_hum = np.load('../../color_color/data_output/x_hum.npy')
y_hum = np.load('../../color_color/data_output/y_hum.npy')
ra_hum = np.load('../../color_color/data_output/ra_hum.npy')
dec_hum = np.load('../../color_color/data_output/dec_hum.npy')
age_hum = np.load('../../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../../color_color/data_output/ebv_hum.npy')
target_name_hum = np.load('../../color_color/data_output/target_name_hum.npy')
clcl_color_hum = np.load('../../color_color/data_output/clcl_color_hum.npy')

color_vi_ml = np.load('../../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../../color_color/data_output/color_ub_ml.npy')
detect_u_ml = np.load('../../color_color/data_output/detect_u_ml.npy')
detect_b_ml = np.load('../../color_color/data_output/detect_b_ml.npy')
detect_v_ml = np.load('../../color_color/data_output/detect_v_ml.npy')
detect_i_ml = np.load('../../color_color/data_output/detect_i_ml.npy')
x_ml = np.load('../../color_color/data_output/x_ml.npy')
y_ml = np.load('../../color_color/data_output/y_ml.npy')
ra_ml = np.load('../../color_color/data_output/ra_ml.npy')
dec_ml = np.load('../../color_color/data_output/dec_ml.npy')
age_ml = np.load('../../color_color/data_output/age_ml.npy')
ebv_ml = np.load('../../color_color/data_output/ebv_ml.npy')
target_name_ml = np.load('../../color_color/data_output/target_name_ml.npy')
clcl_color_ml = np.load('../../color_color/data_output/clcl_color_ml.npy')
detectable_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml


model_ub_sol = np.load('../../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../../color_color/data_output/model_vi_sol.npy')


# categorize data
# get groups of the hull
# convex_hull_gc_ml = fits.open('../../color_color_regions/data_output/convex_hull_ubvi_gc_ml_12.fits')[1].data
# convex_hull_cascade_ml = fits.open('../../color_color_regions/data_output/convex_hull_ubvi_cascade_ml_12.fits')[1].data
# convex_hull_young_ml = fits.open('../../color_color_regions/data_output/convex_hull_ubvi_young_ml_12.fits')[1].data


vi_hull_ogc_ubvi_hum_1 = np.load('../../segmentation/data_output/vi_hull_ogc_ubvi_hum_1.npy')
ub_hull_ogc_ubvi_hum_1 = np.load('../../segmentation/data_output/ub_hull_ogc_ubvi_hum_1.npy')
vi_hull_ogc_ubvi_ml_1 = np.load('../../segmentation/data_output/vi_hull_ogc_ubvi_ml_1.npy')
ub_hull_ogc_ubvi_ml_1 = np.load('../../segmentation/data_output/ub_hull_ogc_ubvi_ml_1.npy')

vi_hull_mid_ubvi_hum_1 = np.load('../../segmentation/data_output/vi_hull_mid_ubvi_hum_1.npy')
ub_hull_mid_ubvi_hum_1 = np.load('../../segmentation/data_output/ub_hull_mid_ubvi_hum_1.npy')
vi_hull_mid_ubvi_ml_1 = np.load('../../segmentation/data_output/vi_hull_mid_ubvi_ml_1.npy')
ub_hull_mid_ubvi_ml_1 = np.load('../../segmentation/data_output/ub_hull_mid_ubvi_ml_1.npy')

vi_hull_young_ubvi_hum_3 = np.load('../../segmentation/data_output/vi_hull_young_ubvi_hum_3.npy')
ub_hull_young_ubvi_hum_3 = np.load('../../segmentation/data_output/ub_hull_young_ubvi_hum_3.npy')
vi_hull_young_ubvi_ml_3 = np.load('../../segmentation/data_output/vi_hull_young_ubvi_ml_3.npy')
ub_hull_young_ubvi_ml_3 = np.load('../../segmentation/data_output/ub_hull_young_ubvi_ml_3.npy')


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


    hdu_rgb_img_hst = fits.open('/media/benutzer/Sicherung/data/phangs_hst/rgb_img/%s_BVI_RGB.fits' % (galaxy_name))
    rgb_img_hst_data_r = plotting_tools.reproject_image(data=hdu_rgb_img_hst[0].data[0],
                                                            wcs=WCS(hdu_rgb_img_hst[0].header, naxis=2),
                                                            new_wcs=wcs_hist, new_shape=(len(x_bins), len(y_bins)))
    rgb_img_hst_data_g = plotting_tools.reproject_image(data=hdu_rgb_img_hst[0].data[1],
                                                            wcs=WCS(hdu_rgb_img_hst[0].header, naxis=2),
                                                            new_wcs=wcs_hist, new_shape=(len(x_bins), len(y_bins)))
    rgb_img_hst_data_b = plotting_tools.reproject_image(data=hdu_rgb_img_hst[0].data[2],
                                                            wcs=WCS(hdu_rgb_img_hst[0].header, naxis=2),
                                                            new_wcs=wcs_hist, new_shape=(len(x_bins), len(y_bins)))
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


    return hst_rgb_image


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


fig, ax = plt.subplots(5, 8, sharex=True, sharey=True, figsize=(20, 13))
fontsize = 18

row_index = 0
col_index = 0
for index in range(len(galaxy_name_list)):
    print(galaxy_name_list[index])
    rgb_img_hst = get_image_data(galaxy_name=galaxy_name_list[index], n_bins=n_bins,
                                 add_side_pixel_frac=add_side_pixel_frac, kernel_size_prop=kernel_size_prop)
    ax[row_index, col_index].imshow(rgb_img_hst, origin='lower')
    ax[row_index, col_index].axis('off')
    ax[row_index, col_index].set_title(galaxy_name_list[index].upper(), fontsize=fontsize)

    col_index += 1
    if col_index == 8:
        row_index += 1
        col_index = 0

ax[4, 6].axis('off')
ax[4, 7].axis('off')

fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97    , wspace=0.01, hspace=0.15)

plt.savefig('plot_output/galaxy_panel.png')
# plt.savefig('plot_output/galaxy_panel.pdf')
plt.cla()
plt.close()
plt.clf()


