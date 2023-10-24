import os.path

import pyregion
import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
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


def get_stellar_assoc(target, img_wcs, res=8):

    if (target[:3] == 'ngc') & (target[3] == 0):
        region_target = target[:3] + target[4:]
    else:
        region_target = target
    if (target == 'ngc0628e') | (target == 'ngc0628c'):
        region_target = 'ngc628'
    if target == 'ngc0685':
        region_target = 'ngc685'
    # open region file
    region_file_path = ('/home/benutzer/data/PHANGS_products/HST_catalogs/IR4/stellar_associations/'
                        '%s/vselect/ws%ipc/') % (region_target, res)
    region_file_name = 'PHANGS_IR4_hst_wfc3_%s_v1p3_multi_assoc-vselect-ws%ipc-region.reg' % (region_target, res)
    if not os.path.isfile(region_file_path + region_file_name):
        return None
    regions = open(region_file_path + region_file_name)
    lines = regions.readlines()

    polygon_list = []
    for line in lines[1:]:
        coordinate_str = line[9:]
        coordinate_str = coordinate_str.replace('d ', ', ')
        coordinate_str = coordinate_str.replace('d', '')
        coord_list = [float(k) for k in coordinate_str.split(', ')]
        ra = coord_list[0::2]
        dec = coord_list[1::2]
        coords_world = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        coords_pix = img_wcs.world_to_pixel(coords_world)
        polygon = Polygon(np.array([coords_pix[0], coords_pix[1]]).T)
        polygon_list.append(polygon)

    return polygon_list


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
# get target list
target_list = catalog_access.target_hst_cc
# load class 3 obejects
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


containing_bool_array_hum_8 = np.array([], dtype=bool)
containing_bool_array_hum_16 = np.array([], dtype=bool)
containing_bool_array_hum_32 = np.array([], dtype=bool)
containing_bool_array_hum_64 = np.array([], dtype=bool)
n_cluster_hum_8 = 0
n_cluster_hum_16 = 0
n_cluster_hum_32 = 0
n_cluster_hum_64 = 0

containing_bool_array_ml_8 = np.array([], dtype=bool)
containing_bool_array_ml_16 = np.array([], dtype=bool)
containing_bool_array_ml_32 = np.array([], dtype=bool)
containing_bool_array_ml_64 = np.array([], dtype=bool)
n_cluster_ml_8 = 0
n_cluster_ml_16 = 0
n_cluster_ml_32 = 0
n_cluster_ml_64 = 0




for target in target_list:
    print('target ', target)
    # getting points
    x_hum_cl3, y_hum_cl3 = catalog_access.get_hst_cc_coords_pix(target=target, cluster_class='class3')
    x_ml_cl3, y_ml_cl3 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml', cluster_class='class3')

    point_list_ml = []
    for x, y in zip(x_ml_cl3, y_ml_cl3):
        point_list_ml.append(Point(x, y))
    point_list_hum = []
    for x, y in zip(x_hum_cl3, y_hum_cl3):
        point_list_hum.append(Point(x, y))

    # get v-band data
    visualization_access = PhotVisualize(
                                        target_name=target,
                                        hst_data_path=hst_data_path,
                                        nircam_data_path=nircam_data_path,
                                        miri_data_path=miri_data_path,
                                        hst_data_ver=hst_data_ver,
                                        nircam_data_ver=nircam_data_ver,
                                        miri_data_ver=miri_data_ver)
    visualization_access.load_hst_nircam_miri_bands(flux_unit='MJy/sr', load_err=False, band_list=['F555W'])
    img_wcs_v_band = visualization_access.hst_bands_data['F555W_wcs_img']

    polygon_list_8 = get_stellar_assoc(target, img_wcs=img_wcs_v_band, res=8)
    if polygon_list_8 is None:
        print(target,' has no 8pc stellar associations')
    else:
        containing_mask_hum_8 = np.zeros(len(x_hum_cl3), dtype=bool)
        containing_mask_ml_8 = np.zeros(len(x_ml_cl3), dtype=bool)

        for polygon in polygon_list_8:
            containing_mask_hum_8 += polygon.contains(point_list_hum)
            containing_mask_ml_8 += polygon.contains(point_list_ml)
        containing_bool_array_hum_8 = np.concatenate([containing_bool_array_hum_8, containing_mask_hum_8])
        containing_bool_array_ml_8 = np.concatenate([containing_bool_array_ml_8, containing_mask_ml_8])
        n_cluster_hum_8 += len(point_list_hum)
        n_cluster_ml_8 += len(point_list_ml)

    polygon_list_16 = get_stellar_assoc(target, img_wcs=img_wcs_v_band, res=16)
    if polygon_list_16 is None:
        print(target,' has no 16pc stellar associations')
    else:
        containing_mask_hum_16 = np.zeros(len(x_hum_cl3), dtype=bool)
        containing_mask_ml_16 = np.zeros(len(x_ml_cl3), dtype=bool)

        for polygon in polygon_list_16:
            containing_mask_hum_16 += polygon.contains(point_list_hum)
            containing_mask_ml_16 += polygon.contains(point_list_ml)
        containing_bool_array_hum_16 = np.concatenate([containing_bool_array_hum_16, containing_mask_hum_16])
        containing_bool_array_ml_16 = np.concatenate([containing_bool_array_ml_16, containing_mask_ml_16])
        n_cluster_hum_16 += len(point_list_hum)
        n_cluster_ml_16 += len(point_list_ml)

    polygon_list_32 = get_stellar_assoc(target, img_wcs=img_wcs_v_band, res=32)
    if polygon_list_32 is None:
        print(target,' has no 32pc stellar associations')
    else:
        containing_mask_hum_32 = np.zeros(len(x_hum_cl3), dtype=bool)
        containing_mask_ml_32 = np.zeros(len(x_ml_cl3), dtype=bool)

        for polygon in polygon_list_32:
            containing_mask_hum_32 += polygon.contains(point_list_hum)
            containing_mask_ml_32 += polygon.contains(point_list_ml)
        containing_bool_array_hum_32 = np.concatenate([containing_bool_array_hum_32, containing_mask_hum_32])
        containing_bool_array_ml_32 = np.concatenate([containing_bool_array_ml_32, containing_mask_ml_32])
        n_cluster_hum_32 += len(point_list_hum)
        n_cluster_ml_32 += len(point_list_ml)

    polygon_list_64 = get_stellar_assoc(target, img_wcs=img_wcs_v_band, res=64)
    if polygon_list_64 is None:
        print(target,' has no 64pc stellar associations')
    else:
        containing_mask_hum_64 = np.zeros(len(x_hum_cl3), dtype=bool)
        containing_mask_ml_64 = np.zeros(len(x_ml_cl3), dtype=bool)

        for polygon in polygon_list_64:
            containing_mask_hum_64 += polygon.contains(point_list_hum)
            containing_mask_ml_64 += polygon.contains(point_list_ml)
        containing_bool_array_hum_64 = np.concatenate([containing_bool_array_hum_64, containing_mask_hum_64])
        containing_bool_array_ml_64 = np.concatenate([containing_bool_array_ml_64, containing_mask_ml_64])
        n_cluster_hum_64 += len(point_list_hum)
        n_cluster_ml_64 += len(point_list_ml)

print(len(containing_bool_array_hum_8))
print(n_cluster_hum_8)
print(len(containing_bool_array_hum_16))
print(n_cluster_hum_16)
print(len(containing_bool_array_hum_32))
print(n_cluster_hum_32)
print(len(containing_bool_array_hum_64))
print(n_cluster_hum_64)
print(len(containing_bool_array_ml_8))
print(n_cluster_ml_8)
print(len(containing_bool_array_ml_16))
print(n_cluster_ml_16)
print(len(containing_bool_array_ml_32))
print(n_cluster_ml_32)
print(len(containing_bool_array_ml_64))
print(n_cluster_ml_64)

print('8pc hum ', sum(containing_bool_array_hum_8), ' of ', n_cluster_hum_8, ' cl3 are in stellar ass. (%.1f) ' %
      ((sum(containing_bool_array_hum_8) / n_cluster_hum_8) * 100))
print('8pc ml ', sum(containing_bool_array_ml_8), ' of ', n_cluster_ml_8, ' cl3 are in stellar ass. (%.1f) ' %
      ((sum(containing_bool_array_ml_8) / n_cluster_ml_8) * 100))

print('16pc hum ', sum(containing_bool_array_hum_16), ' of ', n_cluster_hum_16, ' cl3 are in stellar ass. (%.1f) ' %
      ((sum(containing_bool_array_hum_16) / n_cluster_hum_16) * 100))
print('16pc ml ', sum(containing_bool_array_ml_16), ' of ', n_cluster_ml_16, ' cl3 are in stellar ass. (%.1f) ' %
      ((sum(containing_bool_array_ml_16) / n_cluster_ml_16) * 100))

print('32pc hum ', sum(containing_bool_array_hum_32), ' of ', n_cluster_hum_32, ' cl3 are in stellar ass. (%.1f) ' %
      ((sum(containing_bool_array_hum_32) / n_cluster_hum_32) * 100))
print('32pc ml ', sum(containing_bool_array_ml_32), ' of ', n_cluster_ml_32, ' cl3 are in stellar ass. (%.1f) ' %
      ((sum(containing_bool_array_ml_32) / n_cluster_ml_32) * 100))

print('64pc hum ', sum(containing_bool_array_hum_64), ' of ', n_cluster_hum_64, ' cl3 are in stellar ass. (%.1f) ' %
      ((sum(containing_bool_array_hum_64) / n_cluster_hum_64) * 100))
print('64pc ml ', sum(containing_bool_array_ml_64), ' of ', n_cluster_ml_64, ' cl3 are in stellar ass. (%.1f) ' %
      ((sum(containing_bool_array_ml_64) / n_cluster_ml_64) * 100))


exit()