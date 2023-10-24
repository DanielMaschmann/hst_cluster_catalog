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
nircam_data_ver = 'v0p9p2'
miri_data_ver = 'v0p9p2'

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

target = 'ngc4535'
target_name = 'NGC 4535'
cluster_class = 'ML'
size_of_cutout = (120, 120)
gamma = 3.0


# loading hst images
visualization_access = PhotVisualize(
                                    target_name=target,
                                    hst_data_path=hst_data_path,
                                    nircam_data_path=nircam_data_path,
                                    miri_data_path=miri_data_path,
                                    hst_data_ver=hst_data_ver,
                                    nircam_data_ver=nircam_data_ver,
                                    miri_data_ver=miri_data_ver)
band_list = ['F438W', 'F555W', 'F814W', 'F200W', 'F335M', 'F360M']
visualization_access.load_hst_nircam_miri_bands(flux_unit='MJy/sr', load_err=False, band_list=band_list)

img_data_v = visualization_access.hst_bands_data['F555W_data_img']
img_wcs_v = visualization_access.hst_bands_data['F555W_wcs_img']

center_of_cutout_pix = np.array([img_data_v.shape[0]/2, img_data_v.shape[1]/2])
center_of_cutout_coords = img_wcs_v.pixel_to_world(center_of_cutout_pix[0], center_of_cutout_pix[1])

# get the cutout:
# center_of_cutout = center_of_cutout_coords
center_of_cutout = SkyCoord('12h34m20.2s 8d11m53s', unit=(u.hourangle, u.deg))

# get cutout and produce rgb image
cutout_dict = visualization_access.get_band_cutout_dict(ra_cutout=center_of_cutout.ra, dec_cutout=center_of_cutout.dec,
                                                        cutout_size=size_of_cutout, band_list=band_list)

rgb_cutout_hst = visualization_access.get_rgb_img(data_r=cutout_dict['F814W_img_cutout'].data,
                                              data_g=cutout_dict['F555W_img_cutout'].data,
                                              data_b=cutout_dict['F438W_img_cutout'].data,
                                              min_max_r=(0., 99.9),
                                              min_max_g=(0., 99.9),
                                              min_max_b=(0., 99.9),
                                              gamma_r=3.5, gamma_g=3.5, gamma_b=3.5,
                                              gamma_corr_r=3.5, gamma_corr_g=3.5, gamma_corr_b=3.5, combined_gamma=2.2)

rgb_wcs_hst = cutout_dict['F555W_img_cutout'].wcs
rgb_slice_shape_hst = cutout_dict['F555W_img_cutout'].data.shape

# print(rgb_cutout_hst)
# print(rgb_cutout_hst.shape)
mask_no_signal = ((rgb_cutout_hst[:, :, 0] == rgb_cutout_hst[5, 5, 0]) &
                  (rgb_cutout_hst[:, :, 1] == rgb_cutout_hst[5, 5, 1]) &
                  (rgb_cutout_hst[:, :, 2] == rgb_cutout_hst[5, 5, 2]))
rgb_cutout_hst[mask_no_signal] = 1.0



# new_wcs = cutout_dict['F360M_img_cutout'].wcs
# new_shape = cutout_dict['F360M_img_cutout'].data.shape

new_wcs = rgb_wcs_hst
new_shape = rgb_slice_shape_hst

nircam_img_r = plotting_tools.reproject_image(data=cutout_dict['F360M_img_cutout'].data,
                                              wcs=cutout_dict['F360M_img_cutout'].wcs,
                                              new_wcs=new_wcs,
                                              new_shape=new_shape)
nircam_img_g = plotting_tools.reproject_image(data=cutout_dict['F335M_img_cutout'].data,
                                              wcs=cutout_dict['F335M_img_cutout'].wcs,
                                              new_wcs=new_wcs,
                                              new_shape=new_shape)
nircam_img_b = plotting_tools.reproject_image(data=cutout_dict['F200W_img_cutout'].data,
                                              wcs=cutout_dict['F200W_img_cutout'].wcs,
                                              new_wcs=new_wcs,
                                              new_shape=new_shape)

rgb_cutout_nircam = visualization_access.get_rgb_img(data_r=nircam_img_r,
                                              data_g=nircam_img_g,
                                              data_b=nircam_img_b,
                                              min_max_r=(0., 99.9),
                                              min_max_g=(0., 99.9),
                                              min_max_b=(0., 99.9),
                                              gamma_r=2.2, gamma_g=2.2, gamma_b=2.2,
                                              gamma_corr_r=2.2, gamma_corr_g=2.2, gamma_corr_b=2.2, combined_gamma=2.2)

mask_no_signal = ((rgb_cutout_nircam[:, :, 0] == rgb_cutout_nircam[5, 5, 0]) &
                  (rgb_cutout_nircam[:, :, 1] == rgb_cutout_nircam[5, 5, 1]) &
                  (rgb_cutout_nircam[:, :, 2] == rgb_cutout_nircam[5, 5, 2]))
rgb_cutout_nircam[mask_no_signal] = 1.0


catalog_access.load_hst_cc_list(target_list=[target])
catalog_access.load_hst_cc_list(target_list=[target], cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=[target], classify='ml')
catalog_access.load_hst_cc_list(target_list=[target], classify='ml', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=[target], classify='', cluster_class='candidates')

class_hum_hum_cl12 = catalog_access.get_hst_cc_class_human(target=target)
class_vgg_hum_cl12 = catalog_access.get_hst_cc_class_ml_vgg(target=target)
ra_hum_cl12, dec_hum_cl12 = catalog_access.get_hst_cc_coords_world(target=target)
coords_world_hum_cl12 = SkyCoord(ra=ra_hum_cl12*u.deg, dec=dec_hum_cl12*u.deg)
coords_in_cutout_pix_hum_cl12 = rgb_wcs_hst.world_to_pixel(coords_world_hum_cl12)
mask_in_cutout_hum_cl12 = ((coords_in_cutout_pix_hum_cl12[0] > 0) & (coords_in_cutout_pix_hum_cl12[1] > 0) &
                           (coords_in_cutout_pix_hum_cl12[0] < rgb_slice_shape_hst[0]) &
                           (coords_in_cutout_pix_hum_cl12[1] < rgb_slice_shape_hst[1]))


class_hum_hum_cl3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
class_vgg_hum_cl3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, cluster_class='class3')
ra_hum_cl3, dec_hum_cl3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
coords_world_hum_cl3 = SkyCoord(ra=ra_hum_cl3*u.deg, dec=dec_hum_cl3*u.deg)
coords_in_cutout_pix_hum_cl3 = rgb_wcs_hst.world_to_pixel(coords_world_hum_cl3)
mask_in_cutout_hum_cl3 = ((coords_in_cutout_pix_hum_cl3[0] > 0) & (coords_in_cutout_pix_hum_cl3[1] > 0) &
                          (coords_in_cutout_pix_hum_cl3[0] < rgb_slice_shape_hst[0]) &
                          (coords_in_cutout_pix_hum_cl3[1] < rgb_slice_shape_hst[1]))

class_hum_ml_cl12 = catalog_access.get_hst_cc_class_human(target=target, classify='ml')
class_vgg_ml_cl12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
ra_ml_cl12, dec_ml_cl12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
coords_world_ml_cl12 = SkyCoord(ra=ra_ml_cl12*u.deg, dec=dec_ml_cl12*u.deg)
coords_in_cutout_pix_ml_cl12 = rgb_wcs_hst.world_to_pixel(coords_world_ml_cl12)
mask_in_cutout_ml_cl12 = ((coords_in_cutout_pix_ml_cl12[0] > 0) & (coords_in_cutout_pix_ml_cl12[1] > 0) &
                          (coords_in_cutout_pix_ml_cl12[0] < rgb_slice_shape_hst[0]) &
                          (coords_in_cutout_pix_ml_cl12[1] < rgb_slice_shape_hst[1]))

class_hum_ml_cl3 = catalog_access.get_hst_cc_class_human(target=target, classify='ml', cluster_class='class3')
class_vgg_ml_cl3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
ra_ml_cl3, dec_ml_cl3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
coords_world_ml_cl3 = SkyCoord(ra=ra_ml_cl3*u.deg, dec=dec_ml_cl3*u.deg)
coords_in_cutout_pix_ml_cl3 = rgb_wcs_hst.world_to_pixel(coords_world_ml_cl3)
mask_in_cutout_ml_cl3 = ((coords_in_cutout_pix_ml_cl3[0] > 0) & (coords_in_cutout_pix_ml_cl3[1] > 0) &
                         (coords_in_cutout_pix_ml_cl3[0] < rgb_slice_shape_hst[0]) &
                         (coords_in_cutout_pix_ml_cl3[1] < rgb_slice_shape_hst[1]))

class_hum_cand = catalog_access.get_hst_cc_class_human(target=target, classify='', cluster_class='candidates')
class_vgg_cand = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='', cluster_class='candidates')
ra_cand, dec_cand = catalog_access.get_hst_cc_coords_world(target=target, classify='', cluster_class='candidates')
coords_world_cand = SkyCoord(ra=ra_cand*u.deg, dec=dec_cand*u.deg)
coords_in_cutout_pix_cand = rgb_wcs_hst.world_to_pixel(coords_world_cand)
mask_in_cutout_cand = ((coords_in_cutout_pix_cand[0] > 0) & (coords_in_cutout_pix_cand[1] > 0) &
                       (coords_in_cutout_pix_cand[0] < rgb_slice_shape_hst[0]) &
                       (coords_in_cutout_pix_cand[1] < rgb_slice_shape_hst[1]))



figure = plt.figure(figsize=(30, 16))
fontsize = 30
ax_img_hst = figure.add_axes([0.04, 0.035, 0.47, 0.935], projection=cutout_dict['F555W_img_cutout'].wcs)
ax_img_nircam = figure.add_axes([0.52, 0.035, 0.47, 0.935], projection=new_wcs)

ax_img_hst.imshow(rgb_cutout_hst)
ax_img_nircam.imshow(rgb_cutout_nircam)

if cluster_class == 'Human':
    ax_img_hst.scatter(coords_in_cutout_pix_hum_cl12[0][mask_in_cutout_hum_cl12],
                   coords_in_cutout_pix_hum_cl12[1][mask_in_cutout_hum_cl12],
                   color='white', marker='o', s=280, linewidth=2, facecolor='None')
    ax_img_nircam.scatter(coords_in_cutout_pix_hum_cl12[0][mask_in_cutout_hum_cl12],
                   coords_in_cutout_pix_hum_cl12[1][mask_in_cutout_hum_cl12],
                   color='white', marker='o', s=280, linewidth=2, facecolor='None')
else:
    ax_img_hst.scatter(coords_in_cutout_pix_ml_cl12[0][mask_in_cutout_ml_cl12],
                   coords_in_cutout_pix_ml_cl12[1][mask_in_cutout_ml_cl12],
                   color='white', marker='o', s=280, linewidth=2, facecolor='None')
    ax_img_nircam.scatter(coords_in_cutout_pix_ml_cl12[0][mask_in_cutout_ml_cl12],
                   coords_in_cutout_pix_ml_cl12[1][mask_in_cutout_ml_cl12],
                   color='white', marker='o', s=280, linewidth=2, facecolor='None')

pe = [patheffects.withStroke(linewidth=3, foreground="w")]
ax_img_hst.text(0.03, 0.97, 'HST', horizontalalignment='left', verticalalignment='top',
                color='k', fontsize=fontsize, transform=ax_img_hst.transAxes, path_effects=pe)
ax_img_hst.text(0.03, 0.94, 'F438W', horizontalalignment='left', verticalalignment='top',
                color='blue', fontsize=fontsize, transform=ax_img_hst.transAxes, path_effects=pe)
ax_img_hst.text(0.03, 0.91, 'F555W', horizontalalignment='left', verticalalignment='top',
                color='green', fontsize=fontsize, transform=ax_img_hst.transAxes, path_effects=pe)
ax_img_hst.text(0.03, 0.88, 'F814W', horizontalalignment='left', verticalalignment='top',
                color='red', fontsize=fontsize, transform=ax_img_hst.transAxes, path_effects=pe)

ax_img_nircam.text(0.03, 0.97, 'JWST NIRCAM', horizontalalignment='left', verticalalignment='top',
                color='k', fontsize=fontsize, transform=ax_img_nircam.transAxes, path_effects=pe)
ax_img_nircam.text(0.03, 0.94, 'F200W', horizontalalignment='left', verticalalignment='top',
                color='blue', fontsize=fontsize, transform=ax_img_nircam.transAxes, path_effects=pe)
ax_img_nircam.text(0.03, 0.91, r'F335M (3.3$\mu$ PAH)', horizontalalignment='left', verticalalignment='top',
                color='green', fontsize=fontsize, transform=ax_img_nircam.transAxes, path_effects=pe)
ax_img_nircam.text(0.03, 0.88, 'F360M', horizontalalignment='left', verticalalignment='top',
                color='red', fontsize=fontsize, transform=ax_img_nircam.transAxes, path_effects=pe)



plotting_tools.arr_axis_params(ax=ax_img_hst, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=6, dec_tick_num=5)

plotting_tools.arr_axis_params(ax=ax_img_nircam, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='k',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=6, dec_tick_num=5)

ax_img_hst.set_title('HST ' + target_name + '  (' + cluster_class + ') Class 1+2' , fontsize=fontsize+10)
ax_img_nircam.set_title('JWST NIRCAM ' + target_name + '  (' + cluster_class + ') Class 1+2' , fontsize=fontsize+10)

plt.savefig('plot_output/cluster_%s_%s_hst_nircam.png' % (target, cluster_class))
plt.savefig('plot_output/cluster_%s_%s_hst_nircam.pdf' % (target, cluster_class))
# plt.show()

