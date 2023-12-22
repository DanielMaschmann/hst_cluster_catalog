import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools

from photometry_tools import helper_func as hf, plotting_tools

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.spatial import ConvexHull
import multicolorfits as mcf


red_color_hst = '#FF4433'
green_color_hst = '#0FFF50'
blue_color_hst = '#1F51FF'

def get_rgb_cutout(galaxy_name, pos, cutout_size=(4, 4)):

    hdu_rgb_img_hst = fits.open('/media/benutzer/Sicherung/data/phangs_hst/rgb_img/%s_BVI_RGB.fits' % (galaxy_name))

    cutout_r = hf.get_img_cutout(img=hdu_rgb_img_hst[0].data[0],
                                 wcs=WCS(hdu_rgb_img_hst[0].header, naxis=2), coord=pos, cutout_size=cutout_size)
    cutout_g = hf.get_img_cutout(img=hdu_rgb_img_hst[0].data[1],
                                 wcs=WCS(hdu_rgb_img_hst[0].header, naxis=2), coord=pos, cutout_size=cutout_size)
    cutout_b = hf.get_img_cutout(img=hdu_rgb_img_hst[0].data[2],
                                 wcs=WCS(hdu_rgb_img_hst[0].header, naxis=2), coord=pos, cutout_size=cutout_size)

    # rgb_img_hst_data = np.array([cutout_r.data, cutout_g.data, cutout_b.data]).T
    # return rgb_img_hst_data


    grey_r = mcf.greyRGBize_image(cutout_r.data, rescalefn='asinh', scaletype='perc', min_max=[0.01, 99.5],
                                      gamma=17.5, checkscale=False)
    grey_g = mcf.greyRGBize_image(cutout_g.data, rescalefn='asinh', scaletype='perc', min_max=[0.01, 99.5],
                                      gamma=17.5, checkscale=False)
    grey_b = mcf.greyRGBize_image(cutout_b.data, rescalefn='asinh', scaletype='perc', min_max=[0.01, 99.5],
                                      gamma=17.5, checkscale=False)
    r = mcf.colorize_image(grey_r, red_color_hst, colorintype='hex', gammacorr_color=7.5)
    g = mcf.colorize_image(grey_g, green_color_hst, colorintype='hex', gammacorr_color=7.5)
    b = mcf.colorize_image(grey_b, blue_color_hst, colorintype='hex', gammacorr_color=7.5)
    hst_rgb_image = mcf.combine_multicolor([r, g, b], gamma=7.5, inverse=False)

    return hst_rgb_image


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
mass_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
    mass_list.append(catalog_access.get_target_mstar(target=target))

sort = np.argsort(mass_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]
mass_list = np.array(mass_list)[sort]

count = 0
for index, target in enumerate(target_list):
    if count == 13:
        print('-------------------------')
        count = 0
    print(target)
    count += 1

exit()

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


# vi_hull_ogc_ubvi_hum_1 = np.load('../segmentation/data_output/vi_hull_ogc_ubvi_hum_1.npy')
# ub_hull_ogc_ubvi_hum_1 = np.load('../segmentation/data_output/ub_hull_ogc_ubvi_hum_1.npy')
vi_hull_ogc_ubvi_ml_1 = np.load('../segmentation/data_output/vi_hull_ogc_ubvi_ml_1.npy')
ub_hull_ogc_ubvi_ml_1 = np.load('../segmentation/data_output/ub_hull_ogc_ubvi_ml_1.npy')

# vi_hull_mid_ubvi_hum_1 = np.load('../segmentation/data_output/vi_hull_mid_ubvi_hum_1.npy')
# ub_hull_mid_ubvi_hum_1 = np.load('../segmentation/data_output/ub_hull_mid_ubvi_hum_1.npy')
vi_hull_mid_ubvi_ml_1 = np.load('../segmentation/data_output/vi_hull_mid_ubvi_ml_1.npy')
ub_hull_mid_ubvi_ml_1 = np.load('../segmentation/data_output/ub_hull_mid_ubvi_ml_1.npy')

# vi_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/vi_hull_young_ubvi_hum_3.npy')
# ub_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/ub_hull_young_ubvi_hum_3.npy')
vi_hull_young_ubvi_ml_3 = np.load('../segmentation/data_output/vi_hull_young_ubvi_ml_3.npy')
ub_hull_young_ubvi_ml_3 = np.load('../segmentation/data_output/ub_hull_young_ubvi_ml_3.npy')

hull_ogcc_ml = ConvexHull(np.array([vi_hull_ogc_ubvi_ml_1, ub_hull_ogc_ubvi_ml_1]).T)
hull_map_ml = ConvexHull(np.array([vi_hull_mid_ubvi_ml_1, ub_hull_mid_ubvi_ml_1]).T)
hull_ycl_ml = ConvexHull(np.array([vi_hull_young_ubvi_ml_3, ub_hull_young_ubvi_ml_3]).T)



fig_ycl, ax_ycl = plt.subplots(4, 10, sharex=True, sharey=True, figsize=(20, 8))
fig_map, ax_map = plt.subplots(4, 10, sharex=True, sharey=True, figsize=(20, 8))
fig_ogcc, ax_ogcc = plt.subplots(4, 10, sharex=True, sharey=True, figsize=(20, 8))


cutout_size = (4, 4)
row_index = 0
col_index = 0
for target in target_list:

    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
    dist = catalog_access.dist_dict[galaxy_name]['dist']
    print('target ', target, 'dist ', dist)
    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    v_mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', classify='ml')
    abs_v_mag_ml_12 = hf.conv_mag2abs_mag(mag=v_mag_ml_12, dist=dist)
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')

    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))
    detect_ub_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0))

    in_hull_ogcc_ml = hf.points_in_hull(np.array([color_vi_ml_12, color_ub_ml_12]).T, hull_ogcc_ml)
    in_hull_map_ml = hf.points_in_hull(np.array([color_vi_ml_12, color_ub_ml_12]).T, hull_map_ml)
    in_hull_ycl_ml = hf.points_in_hull(np.array([color_vi_ml_12, color_ub_ml_12]).T, hull_ycl_ml)

    median_abs_v_ml_12_ogcc = np.nanmedian(abs_v_mag_ml_12[in_hull_ogcc_ml])
    median_abs_v_ml_12_map = np.nanmedian(abs_v_mag_ml_12[in_hull_map_ml])
    median_abs_v_ml_12_ycl = np.nanmedian(abs_v_mag_ml_12[in_hull_ycl_ml])

    idx_ogcc = (np.abs(abs_v_mag_ml_12[in_hull_ogcc_ml] - median_abs_v_ml_12_ogcc)).argmin()
    idx_map = (np.abs(abs_v_mag_ml_12[in_hull_map_ml] - median_abs_v_ml_12_map)).argmin()
    idx_ycl = (np.abs(abs_v_mag_ml_12[in_hull_ycl_ml] - median_abs_v_ml_12_ycl)).argmin()

    pos_ogcc = SkyCoord(ra=ra_ml_12[in_hull_ogcc_ml][idx_ogcc]*u.deg, dec=dec_ml_12[in_hull_ogcc_ml][idx_ogcc]*u.deg)
    pos_map = SkyCoord(ra=ra_ml_12[in_hull_map_ml][idx_map]*u.deg, dec=dec_ml_12[in_hull_map_ml][idx_map]*u.deg)
    pos_ycl = SkyCoord(ra=ra_ml_12[in_hull_ycl_ml][idx_ycl]*u.deg, dec=dec_ml_12[in_hull_ycl_ml][idx_ycl]*u.deg)

    rgb_ogcc = get_rgb_cutout(galaxy_name=galaxy_name, pos=pos_ogcc, cutout_size=cutout_size)
    rgb_map = get_rgb_cutout(galaxy_name=galaxy_name, pos=pos_map, cutout_size=cutout_size)
    rgb_ycl = get_rgb_cutout(galaxy_name=galaxy_name, pos=pos_ycl, cutout_size=cutout_size)

    ax_ogcc[row_index, col_index].imshow(rgb_ogcc, origin='lower')
    ax_map[row_index, col_index].imshow(rgb_map, origin='lower')
    ax_ycl[row_index, col_index].imshow(rgb_ycl, origin='lower')

    ax_ogcc[row_index, col_index].axis('off')
    ax_map[row_index, col_index].axis('off')
    ax_ycl[row_index, col_index].axis('off')

    col_index += 1
    if col_index == 10:
        row_index += 1
        col_index = 0



ax_ogcc[3, 9    ].axis('off')
ax_map[3, 9 ].axis('off')
ax_ycl[3, 9 ].axis('off')


fig_ycl.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.001, hspace=0.001)
fig_map.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.001, hspace=0.001)
fig_ogcc.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.001, hspace=0.001)

fig_ycl.savefig('plot_output/median_abs_vmag_ycl.png')
fig_map.savefig('plot_output/median_abs_vmag_map.png')
fig_ogcc.savefig('plot_output/median_abs_vmag_ogcc.png')
