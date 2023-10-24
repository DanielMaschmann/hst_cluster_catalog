import os
import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
from photometry_tools import plotting_tools
from astropy.io import fits
import matplotlib.pyplot as plt
from cluster_cat_dr.visualization_tool import PhotVisualize
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization.wcsaxes import SphericalCircle
from shapely.geometry import Point, MultiPoint
from shapely.geometry.polygon import Polygon
from matplotlib import patheffects




hst_data_path = '/media/benutzer/Sicherung/data/phangs_hst'
nircam_data_path = '/media/benutzer/Sicherung/data/phangs_jwst'
miri_data_path = '/media/benutzer/Sicherung/data/phangs_jwst'
hst_data_ver = 'v1.0'
nircam_data_ver = 'v0p9'
miri_data_ver = 'v0p9'

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path,
                                                            hst_cc_ver='IR4')

target = 'ngc1433'

# loading hst images
visualization_access = PhotVisualize(
                                    target_name=target,
                                    hst_data_path=hst_data_path,
                                    nircam_data_path=nircam_data_path,
                                    miri_data_path=miri_data_path,
                                    hst_data_ver=hst_data_ver,
                                    nircam_data_ver=nircam_data_ver,
                                    miri_data_ver=miri_data_ver)
band_list = ['F438W', 'F555W', 'F814W']
visualization_access.load_hst_nircam_miri_bands(flux_unit='MJy/sr', load_err=False, band_list=band_list)
img_data_b = visualization_access.hst_bands_data['F438W_data_img']
img_wcs_b = visualization_access.hst_bands_data['F438W_wcs_img']
img_data_v = visualization_access.hst_bands_data['F555W_data_img']
img_wcs_v = visualization_access.hst_bands_data['F555W_wcs_img']
img_data_i = visualization_access.hst_bands_data['F814W_data_img']
img_wcs_i = visualization_access.hst_bands_data['F814W_wcs_img']

center_of_cutout_pix = np.array([img_data_v.shape[0]/2, img_data_v.shape[1]/2])
center_of_cutout_coords = img_wcs_v.pixel_to_world(center_of_cutout_pix[0], center_of_cutout_pix[1])


# get the cutout:
size_of_cutout = (200, 200)
# center_of_cutout = SkyCoord('4h19m55.6s -54d55m51s', unit=(u.hourangle, u.deg))
# center_of_cutout = SkyCoord('4h19m56.7s -54d56m5s', unit=(u.hourangle, u.deg))
# center_of_cutout = SkyCoord('4h19m57.5s -54d56m8.5s', unit=(u.hourangle, u.deg))
center_of_cutout = center_of_cutout_coords
# get cutout and produce rgb image
cutout_dict = visualization_access.get_band_cutout_dict(ra_cutout=center_of_cutout.ra, dec_cutout=center_of_cutout.dec,
                                                        cutout_size=size_of_cutout, band_list=band_list)

rgb_cutout = visualization_access.get_rgb_img(data_r=cutout_dict['F814W_img_cutout'].data,
                                              data_g=cutout_dict['F555W_img_cutout'].data,
                                              data_b=cutout_dict['F438W_img_cutout'].data,
                                              min_max_r=(0., 99.95),
                                              min_max_g=(0., 99.95),
                                              min_max_b=(0., 99.95),
                                              gamma_r=3.0, gamma_g=3.0, gamma_b=3.0,
                                              gamma_corr_r=3.0, gamma_corr_g=3.0, gamma_corr_b=3.0, combined_gamma=2.2)
rgb_wcs = cutout_dict['F555W_img_cutout'].wcs
rgb_slice_shape = cutout_dict['F555W_img_cutout'].data.shape

# print(rgb_cutout)
# print(rgb_cutout.shape)
mask_no_signal = ((rgb_cutout[:, :, 0] == rgb_cutout[5, 5, 0]) &
                  (rgb_cutout[:, :, 1] == rgb_cutout[5, 5, 1]) &
                  (rgb_cutout[:, :, 2] == rgb_cutout[5, 5, 2]))
rgb_cutout[mask_no_signal] = 1.0

catalog_access.load_hst_cc_list(target_list=[target])
catalog_access.load_hst_cc_list(target_list=[target], cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=[target], classify='ml')
catalog_access.load_hst_cc_list(target_list=[target], classify='ml', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=[target], classify='', cluster_class='candidates')

class_hum_hum_cl12 = catalog_access.get_hst_cc_class_human(target=target)
class_vgg_hum_cl12 = catalog_access.get_hst_cc_class_ml_vgg(target=target)
ra_hum_cl12, dec_hum_cl12 = catalog_access.get_hst_cc_coords_world(target=target)
coords_world_hum_cl12 = SkyCoord(ra=ra_hum_cl12*u.deg, dec=dec_hum_cl12*u.deg)
coords_in_cutout_pix_hum_cl12 = rgb_wcs.world_to_pixel(coords_world_hum_cl12)
mask_in_cutout_hum_cl12 = ((coords_in_cutout_pix_hum_cl12[0] > 0) & (coords_in_cutout_pix_hum_cl12[1] > 0) &
                           (coords_in_cutout_pix_hum_cl12[0] < rgb_slice_shape[0]) &
                           (coords_in_cutout_pix_hum_cl12[1] < rgb_slice_shape[1]))


class_hum_hum_cl3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
class_vgg_hum_cl3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, cluster_class='class3')
ra_hum_cl3, dec_hum_cl3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
coords_world_hum_cl3 = SkyCoord(ra=ra_hum_cl3*u.deg, dec=dec_hum_cl3*u.deg)
coords_in_cutout_pix_hum_cl3 = rgb_wcs.world_to_pixel(coords_world_hum_cl3)
mask_in_cutout_hum_cl3 = ((coords_in_cutout_pix_hum_cl3[0] > 0) & (coords_in_cutout_pix_hum_cl3[1] > 0) &
                          (coords_in_cutout_pix_hum_cl3[0] < rgb_slice_shape[0]) &
                          (coords_in_cutout_pix_hum_cl3[1] < rgb_slice_shape[1]))




class_hum_ml_cl12 = catalog_access.get_hst_cc_class_human(target=target, classify='ml')
class_vgg_ml_cl12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
ra_ml_cl12, dec_ml_cl12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
coords_world_ml_cl12 = SkyCoord(ra=ra_ml_cl12*u.deg, dec=dec_ml_cl12*u.deg)
coords_in_cutout_pix_ml_cl12 = rgb_wcs.world_to_pixel(coords_world_ml_cl12)
mask_in_cutout_ml_cl12 = ((coords_in_cutout_pix_ml_cl12[0] > 0) & (coords_in_cutout_pix_ml_cl12[1] > 0) &
                          (coords_in_cutout_pix_ml_cl12[0] < rgb_slice_shape[0]) &
                          (coords_in_cutout_pix_ml_cl12[1] < rgb_slice_shape[1]))

class_hum_ml_cl3 = catalog_access.get_hst_cc_class_human(target=target, classify='ml', cluster_class='class3')
class_vgg_ml_cl3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
ra_ml_cl3, dec_ml_cl3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
coords_world_ml_cl3 = SkyCoord(ra=ra_ml_cl3*u.deg, dec=dec_ml_cl3*u.deg)
coords_in_cutout_pix_ml_cl3 = rgb_wcs.world_to_pixel(coords_world_ml_cl3)
mask_in_cutout_ml_cl3 = ((coords_in_cutout_pix_ml_cl3[0] > 0) & (coords_in_cutout_pix_ml_cl3[1] > 0) &
                          (coords_in_cutout_pix_ml_cl3[0] < rgb_slice_shape[0]) &
                          (coords_in_cutout_pix_ml_cl3[1] < rgb_slice_shape[1]))

class_hum_cand = catalog_access.get_hst_cc_class_human(target=target, classify='', cluster_class='candidates')
class_vgg_cand = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='', cluster_class='candidates')
ra_cand, dec_cand = catalog_access.get_hst_cc_coords_world(target=target, classify='', cluster_class='candidates')
coords_world_cand = SkyCoord(ra=ra_cand*u.deg, dec=dec_cand*u.deg)
coords_in_cutout_pix_cand = rgb_wcs.world_to_pixel(coords_world_cand)
mask_in_cutout_cand = ((coords_in_cutout_pix_cand[0] > 0) & (coords_in_cutout_pix_cand[1] > 0) &
                       (coords_in_cutout_pix_cand[0] < rgb_slice_shape[0]) &
                       (coords_in_cutout_pix_cand[1] < rgb_slice_shape[1]))



figure = plt.figure(figsize=(30, 30))
fontsize = 40
ax_img = figure.add_axes([0.05, 0.035, 0.945, 0.93], projection=img_wcs_v)

ax_img.imshow(rgb_cutout)


ax_img.scatter(coords_in_cutout_pix_hum_cl12[0][mask_in_cutout_hum_cl12 * (class_hum_hum_cl12 == 1)],
               coords_in_cutout_pix_hum_cl12[1][mask_in_cutout_hum_cl12 * (class_hum_hum_cl12 == 1)],
               color='tab:red', marker='o', s=280, linewidth=2, facecolor='None')

pe = [patheffects.withStroke(linewidth=3, foreground="w")]
ax_img.text(0.03, 0.97, 'HST', horizontalalignment='left', verticalalignment='top',
                color='k', fontsize=fontsize, transform=ax_img.transAxes, path_effects=pe)
ax_img.text(0.08, 0.97, 'B', horizontalalignment='left', verticalalignment='top',
                color='blue', fontsize=fontsize, transform=ax_img.transAxes, path_effects=pe)
ax_img.text(0.095, 0.97, 'V', horizontalalignment='left', verticalalignment='top',
                color='green', fontsize=fontsize, transform=ax_img.transAxes, path_effects=pe)
ax_img.text(0.11, 0.97, 'I', horizontalalignment='left', verticalalignment='top',
                color='red', fontsize=fontsize, transform=ax_img.transAxes, path_effects=pe)

plotting_tools.arr_axis_params(ax=ax_img, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=6, dec_tick_num=5)

ax_img.set_title('NGC 1433', fontsize=fontsize)

plt.savefig('plot_output/cluster_%s.png' % target)
# plt.show()

