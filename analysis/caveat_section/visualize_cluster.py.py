import os
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

target = 'ngc0628c'
# galaxy_name = 'ngc0682'

# loading hst images
visualization_access = PhotVisualize(
                                    target_name=target,
                                    hst_data_path=hst_data_path,
                                    nircam_data_path=nircam_data_path,
                                    miri_data_path=miri_data_path,
                                    hst_data_ver=hst_data_ver,
                                    nircam_data_ver=nircam_data_ver,
                                    miri_data_ver=miri_data_ver)
band_list = ['F435W', 'F555W', 'F814W']
visualization_access.load_hst_nircam_miri_bands(flux_unit='MJy/sr', load_err=False, band_list=band_list)
# img_data_b = visualization_access.hst_bands_data['F438W_data_img']
# img_wcs_b = visualization_access.hst_bands_data['F438W_wcs_img']
# img_data_v = visualization_access.hst_bands_data['F555W_data_img']
# img_wcs_v = visualization_access.hst_bands_data['F555W_wcs_img']
# img_data_i = visualization_access.hst_bands_data['F814W_data_img']
# img_wcs_i = visualization_access.hst_bands_data['F814W_wcs_img']

# get the cutout:
size_of_cutout = (70, 70)
# center_of_cutout = SkyCoord('4h19m55.6s -54d55m51s', unit=(u.hourangle, u.deg))
# center_of_cutout = SkyCoord('4h19m56.7s -54d56m5s', unit=(u.hourangle, u.deg))
# center_of_cutout = SkyCoord('4h19m57.5s -54d56m8.5s', unit=(u.hourangle, u.deg))
center_of_cutout = SkyCoord('1h36m40.0s 15d46m0.5s', unit=(u.hourangle, u.deg))
# get cutout and produce rgb image
cutout_dict = visualization_access.get_band_cutout_dict(ra_cutout=center_of_cutout.ra, dec_cutout=center_of_cutout.dec,
                                                        cutout_size=size_of_cutout, band_list=band_list)

rgb_cutout = visualization_access.get_rgb_img(data_r=cutout_dict['F814W_img_cutout'].data,
                                              data_g=cutout_dict['F555W_img_cutout'].data,
                                              data_b=cutout_dict['F435W_img_cutout'].data,
                                              min_max_r=(0., 99.8),
                                              min_max_g=(0., 99.8),
                                              min_max_b=(0., 99.8))
rgb_wcs = cutout_dict['F555W_img_cutout'].wcs
rgb_slice_shape = cutout_dict['F555W_img_cutout'].data.shape




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



print(class_hum_cand[mask_in_cutout_cand])
print(class_vgg_cand[mask_in_cutout_cand])

polygon_list_8 = get_stellar_assoc(target, img_wcs=cutout_dict['F555W_img_cutout'].wcs, res=8)
polygon_list_16 = get_stellar_assoc(target, img_wcs=cutout_dict['F555W_img_cutout'].wcs, res=16)
polygon_list_32 = get_stellar_assoc(target, img_wcs=cutout_dict['F555W_img_cutout'].wcs, res=32)
polygon_list_64 = get_stellar_assoc(target, img_wcs=cutout_dict['F555W_img_cutout'].wcs, res=64)


figure = plt.figure(figsize=(30, 30))
fontsize = 40
ax_img = figure.add_axes([0.06, 0.04, 0.93, 0.93], projection=cutout_dict['F555W_img_cutout'].wcs)

ax_img.imshow(rgb_cutout)

# ax_img.scatter(coords_in_cutout_pix_hum_cl3[0][mask_in_cutout_hum_cl3],
#                coords_in_cutout_pix_hum_cl3[1][mask_in_cutout_hum_cl3],
#                color='magenta', s=150)
# ax_img.scatter(coords_in_cutout_pix_ml_cl3[0][mask_in_cutout_ml_cl3],
#                coords_in_cutout_pix_ml_cl3[1][mask_in_cutout_ml_cl3],
#                color='cyan', s=100)
# ax_img.scatter(coords_in_cutout_pix_cand[0][mask_in_cutout_cand * (class_hum_cand > 3)],
#                coords_in_cutout_pix_cand[1][mask_in_cutout_cand * (class_hum_cand > 3)],
#                color='red', )


ax_img.scatter(coords_in_cutout_pix_hum_cl12[0][mask_in_cutout_hum_cl12 * (class_hum_hum_cl12 == 1)],
               coords_in_cutout_pix_hum_cl12[1][mask_in_cutout_hum_cl12 * (class_hum_hum_cl12 == 1)],
               color='tab:red', marker='o', s=280, linewidth=2, facecolor='None')
ax_img.scatter(coords_in_cutout_pix_ml_cl12[0][mask_in_cutout_ml_cl12 * (class_vgg_ml_cl12 == 1)],
               coords_in_cutout_pix_ml_cl12[1][mask_in_cutout_ml_cl12 * (class_vgg_ml_cl12 == 1)],
               color='tab:red', marker='s', s=480, linewidth=2, facecolor='None')

ax_img.scatter(coords_in_cutout_pix_hum_cl12[0][mask_in_cutout_hum_cl12 * (class_hum_hum_cl12 == 2)],
               coords_in_cutout_pix_hum_cl12[1][mask_in_cutout_hum_cl12 * (class_hum_hum_cl12 == 2)],
               color='tab:blue', marker='o', s=280, linewidth=2, facecolor='None')
ax_img.scatter(coords_in_cutout_pix_ml_cl12[0][mask_in_cutout_ml_cl12 * (class_vgg_ml_cl12 == 2)],
               coords_in_cutout_pix_ml_cl12[1][mask_in_cutout_ml_cl12 * (class_vgg_ml_cl12 == 2)],
               color='tab:blue', marker='s', s=480, linewidth=2, facecolor='None')


ax_img.scatter(coords_in_cutout_pix_hum_cl3[0][mask_in_cutout_hum_cl3 * (class_hum_hum_cl3 == 3)],
               coords_in_cutout_pix_hum_cl3[1][mask_in_cutout_hum_cl3 * (class_hum_hum_cl3 == 3)],
               color='tab:green', marker='o', s=280, linewidth=2, facecolor='None')
ax_img.scatter(coords_in_cutout_pix_ml_cl3[0][mask_in_cutout_ml_cl3 * (class_vgg_ml_cl3 == 3)],
               coords_in_cutout_pix_ml_cl3[1][mask_in_cutout_ml_cl3 * (class_vgg_ml_cl3 == 3)],
               color='tab:green', marker='s', s=480, linewidth=2, facecolor='None')



ax_img.scatter(coords_in_cutout_pix_cand[0][mask_in_cutout_cand * (class_hum_cand == 4)],
               coords_in_cutout_pix_cand[1][mask_in_cutout_cand * (class_hum_cand == 4)],
               color='tab:orange', marker='o', s=180, linewidth=2, facecolor='None')
ax_img.scatter(coords_in_cutout_pix_cand[0][mask_in_cutout_cand * (class_hum_cand == 5)],
               coords_in_cutout_pix_cand[1][mask_in_cutout_cand * (class_hum_cand == 5)],
               color='tab:cyan', marker='o', s=180, linewidth=2, facecolor='None')
ax_img.scatter(coords_in_cutout_pix_cand[0][mask_in_cutout_cand * (class_hum_cand == 13)],
               coords_in_cutout_pix_cand[1][mask_in_cutout_cand * (class_hum_cand == 13)],
               color='tab:pink', marker='o', s=180, linewidth=2, facecolor='None')



for polygon in polygon_list_16:
    x, y = polygon.exterior.xy
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask_inside_cutout = ((x > 0) & (y > 0) & (x < cutout_dict['F555W_img_cutout'].data.shape[0]) &
                          (y < cutout_dict['F555W_img_cutout'].data.shape[1]))
    ax_img.plot(x[mask_inside_cutout],y[mask_inside_cutout], color='darkorchid', linewidth=2, linestyle='-')
for polygon in polygon_list_32:
    x, y = polygon.exterior.xy
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask_inside_cutout = ((x > 0) & (y > 0) & (x < cutout_dict['F555W_img_cutout'].data.shape[0]) &
                          (y < cutout_dict['F555W_img_cutout'].data.shape[1]))
    ax_img.plot(x[mask_inside_cutout],y[mask_inside_cutout], color='darkorchid', linewidth=2, linestyle='--')
for polygon in polygon_list_64:
    x, y = polygon.exterior.xy
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask_inside_cutout = ((x > 0) & (y > 0) & (x < cutout_dict['F555W_img_cutout'].data.shape[0]) &
                          (y < cutout_dict['F555W_img_cutout'].data.shape[1]))
    ax_img.plot(x[mask_inside_cutout],y[mask_inside_cutout], color='darkorchid', linewidth=2, linestyle=':')


ax_img.set_xlim(0, rgb_slice_shape[0])
ax_img.set_ylim(0, rgb_slice_shape[1])

plt.savefig('plot_output/cluster_type_vizual.png')


exit()



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
