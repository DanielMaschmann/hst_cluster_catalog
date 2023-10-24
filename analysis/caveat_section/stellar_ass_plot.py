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

target_list = catalog_access.target_hst_cc
# dist_list = []
# for target in target_list:
#     if (target == 'ngc0628c') | (target == 'ngc0628e'):
#         galaxy_name = 'ngc0628'
#     else:
#         galaxy_name = target
#     dist_list.append(catalog_access.dist_dict[galaxy_name]['dist'])
# sort = np.argsort(dist_list)
# target_list = np.array(target_list)[sort]
# dist_list = np.array(dist_list)[sort]
#
# catalog_access.load_hst_cc_list(target_list=target_list)
# catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')
#
target = 'ngc1300'

# class_hum_gal = catalog_access.get_hst_cc_class_human(target=target, classify='ml', cluster_class='class3')
# class_vgg_gal = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
# ra_cand, dec_cand = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
x_ml_cl3, y_ml_cl3 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml', cluster_class='class3')

point_list = []
for x, y in zip(x_ml_cl3, y_ml_cl3):
    point_list.append(Point(x, y))

# print(point_list)
# print(MultiPoint(point_list))
# print(MultiPoint(np.array([x_ml_cl3, y_ml_cl3]).T))
# exit()
#
# print(MultiPoint(np.array([x_ml_cl3, y_ml_cl3]).T).xy)
#
# exit()

visualization_access = PhotVisualize(
                                    target_name=target,
                                    hst_data_path=hst_data_path,
                                    nircam_data_path=nircam_data_path,
                                    miri_data_path=miri_data_path,
                                    hst_data_ver=hst_data_ver,
                                    nircam_data_ver=nircam_data_ver,
                                    miri_data_ver=miri_data_ver)
visualization_access.load_hst_nircam_miri_bands(flux_unit='MJy/sr', load_err=False, band_list=['F555W'])
img_data = visualization_access.hst_bands_data['F555W_data_img']
img_wcs = visualization_access.hst_bands_data['F555W_wcs_img']
img_header = visualization_access.hst_bands_data['F555W_header_img']


figure = plt.figure(figsize=(20, 20))
fontsize = 40
ax_img = figure.add_axes([0.06, 0.04, 0.93, 0.93], projection=img_wcs)

ax_img.imshow(np.log10(img_data), cmap='Greys')# , vmin=0, vmax=10)

# open region file
region_file_path = ('/home/benutzer/data/PHANGS_products/HST_catalogs/IR4/stellar_associations/'
                    '%s/vselect/ws16pc/') % target
region_file_name = 'PHANGS_IR4_hst_wfc3_%s_v1p3_multi_assoc-vselect-ws16pc-region.reg' % target

regions = open(region_file_path + region_file_name)

lines = regions.readlines()


containing_mask = np.zeros(len(x_ml_cl3), dtype=bool)

index = 0
for line in lines[1:]:
    coordinate_str = line[9:]
    coordinate_str = coordinate_str.replace('d ', ', ')
    coordinate_str = coordinate_str.replace('d', '')
    # print('coordinate_str ', coordinate_str)
    coord_list = [float(k) for k in coordinate_str.split(', ')]
    ra = coord_list[0::2]
    dec = coord_list[1::2]
    coords_world = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    coords_pix = img_wcs.world_to_pixel(coords_world)
    polygon = Polygon(np.array([coords_pix[0], coords_pix[1]]).T)
    x,y = polygon.exterior.xy
    # print(np.array([x_ml_cl3, y_ml_cl3]).T)
    # exit()
    containing_mask += polygon.contains(point_list)

    ax_img.plot(x,y, color='r')
    index += 1

ax_img.scatter(x_ml_cl3, y_ml_cl3)
ax_img.scatter(x_ml_cl3[containing_mask], y_ml_cl3[containing_mask])
print(index)
plt.show()
exit()


polygon = ' polygon 170.071709 12.995132 170.071698 12.995132 170.071687 12.995132 170.071675 12.995132 170.071664 12.995132 170.071664 12.995132 170.071653 12.995143 170.071653 12.995143 170.071641 12.995154 170.071641 12.995154 170.071641 12.995165 170.071641 12.995176 170.071641 12.995187 170.071641 12.995198 170.071641 12.995199 170.071653 12.995209 170.071653 12.995210 170.071664 12.995220 170.071664 12.995221 170.071675 12.995221 170.071687 12.995221 170.071698 12.995232 170.071698 12.995232 170.071709 12.995232 170.071721 12.995232 170.071732 12.995243 170.071732 12.995243 170.071743 12.995254 170.071743 12.995254 170.071754 12.995265 170.071754 12.995265 170.071766 12.995265 170.071777 12.995265 170.071788 12.995265 170.071800 12.995265 170.071811 12.995265 170.071822 12.995276 170.071822 12.995276 170.071833 12.995276 170.071845 12.995287 170.071845 12.995287 170.071856 12.995287 170.071867 12.995287 170.071879 12.995287 170.071879 12.995287 170.071890 12.995276 170.071890 12.995276 170.071890 12.995265 170.071890 12.995254 170.071890 12.995253 170.071879 12.995243 170.071879 12.995242 170.071868 12.995231 170.071867 12.995231 170.071856 12.995231 170.071845 12.995231 170.071834 12.995220 170.071833 12.995220 170.071822 12.995209 170.071822 12.995209 170.071811 12.995198 170.071811 12.995198 170.071800 12.995187 170.071800 12.995187 170.071788 12.995187 170.071777 12.995187 170.071766 12.995176 170.071766 12.995176 170.071754 12.995176 170.071743 12.995165 170.071743 12.995165 170.071732 12.995154 170.071732 12.995154 170.071721 12.995143 170.071721 12.995143 170.071709 12.995132 170.071709 12.995132d'
region = pyregion.parse(polygon)
print(region)

exit()





class_hum = np.array([])
class_vgg = np.array([])

ra = np.array([])
dec = np.array([])

for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
    dist = catalog_access.dist_dict[galaxy_name]['dist']
    print('target ', target, 'dist ', dist)

    class_hum_gal = catalog_access.get_hst_cc_class_human(target=target, classify='ml', cluster_class='class3')
    class_vgg_gal = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')

    class_hum = np.concatenate([class_hum, class_hum_gal])
    class_vgg = np.concatenate([class_vgg, class_vgg_gal])

    ra_cand, dec_cand = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    ra = np.concatenate([ra, ra_cand])
    dec = np.concatenate([dec, dec_cand])
