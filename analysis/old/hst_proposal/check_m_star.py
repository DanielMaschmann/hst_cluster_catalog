import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse


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
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
    dist_list.append(catalog_access.dist_dict[galaxy_name]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


mass_hum = np.array([])
clcl_color_hum = np.array([])

mass_ml = np.array([])
clcl_qual_color_ml = np.array([])
clcl_color_ml = np.array([])

for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    mass_hum_12 = catalog_access.get_hst_cc_stellar_m(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    mass_hum_3 = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')

    mass_hum = np.concatenate([mass_hum, mass_hum_12, mass_hum_3])
    clcl_color_hum = np.concatenate([clcl_color_hum, cluster_class_hum_12, cluster_class_hum_3])


    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    mass_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')

    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    mass_ml_3 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')

    mass_ml = np.concatenate([mass_ml, mass_ml_12, mass_ml_3])
    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    clcl_qual_color_ml = np.concatenate([clcl_qual_color_ml, cluster_class_qual_ml_12, cluster_class_qual_ml_3])



mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_class_1_ml = (clcl_color_ml == 1) #& (clcl_qual_color_ml >= 0.9)
mask_class_2_ml = (clcl_color_ml == 2) #& (clcl_qual_color_ml >= 0.9)
mask_class_3_ml = (clcl_color_ml == 3) #& (clcl_qual_color_ml >= 0.9)

print(mass_hum)

print('m star Hum > 5000 Msol = ', sum((mass_hum > 5000) * mask_class_1_hum), ' of ', sum(mask_class_1_hum), ' class 1 clusters ')
print('m star Hum > 5000 Msol = ', sum((mass_hum > 5000) * mask_class_2_hum), ' of ', sum(mask_class_2_hum), ' class 2 clusters ')
print('m star Hum > 5000 Msol = ', sum((mass_hum > 5000) * mask_class_3_hum), ' of ', sum(mask_class_3_hum), ' class 3 clusters ')

print('m star Hum > 10000 Msol = ', sum((mass_hum > 10000) * mask_class_1_hum), ' of ', sum(mask_class_1_hum), ' class 1 clusters ')
print('m star Hum > 10000 Msol = ', sum((mass_hum > 10000) * mask_class_2_hum), ' of ', sum(mask_class_2_hum), ' class 2 clusters ')
print('m star Hum > 10000 Msol = ', sum((mass_hum > 10000) * mask_class_3_hum), ' of ', sum(mask_class_3_hum), ' class 3 clusters ')

print('m star ML > 5000 Msol = ', sum((mass_ml > 5000) * mask_class_1_ml), ' of ', sum(mask_class_1_ml), ' class 1 clusters ')
print('m star ML > 5000 Msol = ', sum((mass_ml > 5000) * mask_class_2_ml), ' of ', sum(mask_class_2_ml), ' class 2 clusters ')
print('m star ML > 5000 Msol = ', sum((mass_ml > 5000) * mask_class_3_ml), ' of ', sum(mask_class_3_ml), ' class 3 clusters ')

print('m star ML > 10000 Msol = ', sum((mass_ml > 10000) * mask_class_1_ml), ' of ', sum(mask_class_1_ml), ' class 1 clusters ')
print('m star ML > 10000 Msol = ', sum((mass_ml > 10000) * mask_class_2_ml), ' of ', sum(mask_class_2_ml), ' class 2 clusters ')
print('m star ML > 10000 Msol = ', sum((mass_ml > 10000) * mask_class_3_ml), ' of ', sum(mask_class_3_ml), ' class 3 clusters ')




exit()
