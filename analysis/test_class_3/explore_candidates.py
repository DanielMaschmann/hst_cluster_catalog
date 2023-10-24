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
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
    dist_list.append(catalog_access.dist_dict[galaxy_name]['dist'])

# catalog_access.load_hst_cc_list(target_list=target_list)
# catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='', cluster_class='candidates')

#
target = 'ngc3627'

ra_cand, dec_cand = catalog_access.get_hst_cc_coords_world(target=target, classify='', cluster_class='candidates')
x_cand, y_cand = catalog_access.get_hst_cc_coords_pix(target=target, classify='', cluster_class='candidates')

# print(len(ra_cand))
# number_close_objs = 0
# pos_all = SkyCoord(ra=ra_cand*u.deg, dec=dec_cand*u.deg)
# for index in range(len(ra_cand)):
#     pos_individual = SkyCoord(ra=ra_cand[index]*u.deg, dec=dec_cand[index]*u.deg)
#     sep = pos_individual.separation(pos_all)
#     if sum(sep < 0.12*u.arcsec) > 1:
#         number_close_objs += 1
#
# print(number_close_objs)
# exit()


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

    class_hum_gal = catalog_access.get_hst_cc_class_human(target=target, classify='', cluster_class='candidates')
    class_vgg_gal = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='', cluster_class='candidates')

    class_hum = np.concatenate([class_hum, class_hum_gal])
    class_vgg = np.concatenate([class_vgg, class_vgg_gal])

    ra_cand, dec_cand = catalog_access.get_hst_cc_coords_world(target=target, classify='', cluster_class='candidates')
    ra = np.concatenate([ra, ra_cand])
    dec = np.concatenate([dec, dec_cand])

mask_hum_cl4 = class_hum > 3

mask_doublet = class_hum == 5
mask_triplets = class_hum == 13


print(sum(mask_hum_cl4))
print(sum(mask_doublet))
print(sum(mask_triplets))

print(sum(mask_doublet) / sum(mask_hum_cl4))
print(sum(mask_triplets) / sum(mask_hum_cl4))

print(sum(mask_triplets + mask_doublet) / sum(mask_hum_cl4))



exit()



#
# print(len(ra))
# number_close_objs = 0
# pos_all = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
# for index in range(len(ra)):
#     pos_individual = SkyCoord(ra=ra[index]*u.deg, dec=dec[index]*u.deg)
#     sep = pos_individual.separation(pos_all)
#     if sum(sep < 0.12*u.arcsec) > 1:
#         number_close_objs += 1
#
# print('number_close_objs', number_close_objs)
# print(number_close_objs / len(ra))



# print(sum(hum_classified))
# print(sum(class_hum == 1))
# print(sum(class_hum == 2))
# print(sum(class_hum == 3))
# print(sum(class_hum > 3))
# print(7857 + 7691 + 6240 + 16555)
# exit()

hum_classified = class_hum > 0
mask_both_classified = (class_hum > 0) & (class_vgg != -999)

# print(sum(hum_classified))
# print(sum(mask_both_classified))
#
# exit()

mask_hum_cl1 = class_hum == 1
mask_hum_cl2 = class_hum == 2
mask_hum_cl3 = class_hum == 3
mask_hum_cl4 = class_hum > 3

mask_vgg_cl1 = class_vgg == 1
mask_vgg_cl2 = class_vgg == 2
mask_vgg_cl3 = class_vgg == 3
mask_vgg_cl4 = class_vgg > 3



print('c1-c1 %.1f' % (sum(mask_hum_cl1*mask_both_classified*mask_vgg_cl1) / sum(mask_hum_cl1*mask_both_classified) * 100))
print('c1-c2 %.1f' % (sum(mask_hum_cl1*mask_both_classified*mask_vgg_cl2) / sum(mask_hum_cl1*mask_both_classified) * 100))
print('c1-c3 %.1f' % (sum(mask_hum_cl1*mask_both_classified*mask_vgg_cl3) / sum(mask_hum_cl1*mask_both_classified) * 100))
print('c1-c4 %.1f' % (sum(mask_hum_cl1*mask_both_classified*mask_vgg_cl4) / sum(mask_hum_cl1*mask_both_classified) * 100))

print('c2-c1 %.1f' % (sum(mask_hum_cl2*mask_both_classified*mask_vgg_cl1) / sum(mask_hum_cl2*mask_both_classified) * 100))
print('c2-c2 %.1f' % (sum(mask_hum_cl2*mask_both_classified*mask_vgg_cl2) / sum(mask_hum_cl2*mask_both_classified) * 100))
print('c2-c3 %.1f' % (sum(mask_hum_cl2*mask_both_classified*mask_vgg_cl3) / sum(mask_hum_cl2*mask_both_classified) * 100))
print('c2-c4 %.1f' % (sum(mask_hum_cl2*mask_both_classified*mask_vgg_cl4) / sum(mask_hum_cl2*mask_both_classified) * 100))

print('c3-c1 %.1f' % (sum(mask_hum_cl3*mask_both_classified*mask_vgg_cl1) / sum(mask_hum_cl3*mask_both_classified) * 100))
print('c3-c2 %.1f' % (sum(mask_hum_cl3*mask_both_classified*mask_vgg_cl2) / sum(mask_hum_cl3*mask_both_classified) * 100))
print('c3-c3 %.1f' % (sum(mask_hum_cl3*mask_both_classified*mask_vgg_cl3) / sum(mask_hum_cl3*mask_both_classified) * 100))
print('c3-c4 %.1f' % (sum(mask_hum_cl3*mask_both_classified*mask_vgg_cl4) / sum(mask_hum_cl3*mask_both_classified) * 100))

print('c4-c1 %.1f' % (sum(mask_hum_cl4*mask_both_classified*mask_vgg_cl1) / sum(mask_hum_cl4*mask_both_classified) * 100))
print('c4-c2 %.1f' % (sum(mask_hum_cl4*mask_both_classified*mask_vgg_cl2) / sum(mask_hum_cl4*mask_both_classified) * 100))
print('c4-c3 %.1f' % (sum(mask_hum_cl4*mask_both_classified*mask_vgg_cl3) / sum(mask_hum_cl4*mask_both_classified) * 100))
print('c4-c4 %.1f' % (sum(mask_hum_cl4*mask_both_classified*mask_vgg_cl4) / sum(mask_hum_cl4*mask_both_classified) * 100))


print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


print(sum(mask_both_classified))
print(sum(mask_vgg_cl1*mask_both_classified))
print(sum(mask_vgg_cl2*mask_both_classified))
print(sum(mask_vgg_cl3*mask_both_classified))
print(sum(mask_vgg_cl4*mask_both_classified))

print('######')
# print(sum(mask_vgg_cl1 * mask_both_classified))
# print(sum(mask_hum_cl1 * mask_vgg_cl1 * mask_both_classified))
# print(sum(mask_hum_cl2 * mask_vgg_cl1 * mask_both_classified))
# print(sum(mask_hum_cl3 * mask_vgg_cl1 * mask_both_classified))
# print(sum(mask_hum_cl4 * mask_vgg_cl1 * mask_both_classified))
# print(5961 + 802 + 74 + 1053)
#
#
# exit()


print('c1-c1 %.1f' % (sum(mask_hum_cl1*mask_vgg_cl1*mask_both_classified) / sum(mask_vgg_cl1*mask_both_classified) * 100))
print('c1-c2 %.1f' % (sum(mask_hum_cl2*mask_vgg_cl1*mask_both_classified) / sum(mask_vgg_cl1*mask_both_classified) * 100))
print('c1-c3 %.1f' % (sum(mask_hum_cl3*mask_vgg_cl1*mask_both_classified) / sum(mask_vgg_cl1*mask_both_classified) * 100))
print('c1-c4 %.1f' % (sum(mask_hum_cl4*mask_vgg_cl1*mask_both_classified) / sum(mask_vgg_cl1*mask_both_classified) * 100))

print('c2-c1 %.1f' % (sum(mask_vgg_cl2 * mask_hum_cl1*mask_both_classified) / sum(mask_vgg_cl2*mask_both_classified) * 100))
print('c2-c2 %.1f' % (sum(mask_vgg_cl2 * mask_hum_cl2*mask_both_classified) / sum(mask_vgg_cl2*mask_both_classified) * 100))
print('c2-c3 %.1f' % (sum(mask_vgg_cl2 * mask_hum_cl3*mask_both_classified) / sum(mask_vgg_cl2*mask_both_classified) * 100))
print('c2-c4 %.1f' % (sum(mask_vgg_cl2 * mask_hum_cl4*mask_both_classified) / sum(mask_vgg_cl2*mask_both_classified) * 100))

print('c3-c1 %.1f' % (sum(mask_vgg_cl3 * mask_hum_cl1*mask_both_classified) / sum(mask_vgg_cl3*mask_both_classified) * 100))
print('c3-c2 %.1f' % (sum(mask_vgg_cl3 * mask_hum_cl2*mask_both_classified) / sum(mask_vgg_cl3*mask_both_classified) * 100))
print('c3-c3 %.1f' % (sum(mask_vgg_cl3 * mask_hum_cl3*mask_both_classified) / sum(mask_vgg_cl3*mask_both_classified) * 100))
print('c3-c4 %.1f' % (sum(mask_vgg_cl3 * mask_hum_cl4*mask_both_classified) / sum(mask_vgg_cl3*mask_both_classified) * 100))

print('c4-c1 %.1f' % (sum(mask_vgg_cl4 * mask_hum_cl1*mask_both_classified) / sum(mask_vgg_cl4*mask_both_classified) * 100))
print('c4-c2 %.1f' % (sum(mask_vgg_cl4 * mask_hum_cl2*mask_both_classified) / sum(mask_vgg_cl4*mask_both_classified) * 100))
print('c4-c3 %.1f' % (sum(mask_vgg_cl4 * mask_hum_cl3*mask_both_classified) / sum(mask_vgg_cl4*mask_both_classified) * 100))
print('c4-c4 %.1f' % (sum(mask_vgg_cl4 * mask_hum_cl4*mask_both_classified) / sum(mask_vgg_cl4*mask_both_classified) * 100))


