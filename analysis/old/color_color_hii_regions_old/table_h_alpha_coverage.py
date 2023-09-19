import os.path

import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from scipy.stats import gaussian_kde
import dust_tools.extinction_tools
from matplotlib.colorbar import ColorbarBase
import matplotlib


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
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')




for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]


    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    cluster_id_hum_12 = catalog_access.get_hst_cc_phangs_id(target)
    cluster_id_hum_3 = catalog_access.get_hst_cc_phangs_id(target, cluster_class='class3')
    class_1_hum = (cluster_class_hum_12 == 1)
    class_2_hum = (cluster_class_hum_12 == 2)
    class_3_hum = (cluster_class_hum_3 == 3)

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    age_ml_3 = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    cluster_id_ml_12 = catalog_access.get_hst_cc_phangs_id(target, classify='ml')
    cluster_id_ml_3 = catalog_access.get_hst_cc_phangs_id(target, classify='ml', cluster_class='class3')
    class_1_ml = (cluster_class_ml_12 == 1)
    class_2_ml = (cluster_class_ml_12 == 2)
    class_3_ml = (cluster_class_ml_3 == 3)

    h_alpha_intensity_hum_12 = np.zeros(len(cluster_id_hum_12))
    hii_reg_hum_12 = np.zeros(len(cluster_id_hum_12))
    hii_reg_3_hum_12 = np.zeros(len(cluster_id_hum_12))
    hii_reg_5_hum_12 = np.zeros(len(cluster_id_hum_12))
    hii_reg_10_hum_12 = np.zeros(len(cluster_id_hum_12))
    h_alpha_intensity_hum_3 = np.zeros(len(cluster_id_hum_3))
    hii_reg_hum_3 = np.zeros(len(cluster_id_hum_3))
    hii_reg_3_hum_3 = np.zeros(len(cluster_id_hum_3))
    hii_reg_5_hum_3 = np.zeros(len(cluster_id_hum_3))
    hii_reg_10_hum_3 = np.zeros(len(cluster_id_hum_3))

    h_alpha_intensity_ml_12 = np.zeros(len(cluster_id_ml_12))
    hii_reg_ml_12 = np.zeros(len(cluster_id_ml_12))
    hii_reg_3_ml_12 = np.zeros(len(cluster_id_ml_12))
    hii_reg_5_ml_12 = np.zeros(len(cluster_id_ml_12))
    hii_reg_10_ml_12 = np.zeros(len(cluster_id_ml_12))
    h_alpha_intensity_ml_3 = np.zeros(len(cluster_id_ml_3))
    hii_reg_ml_3 = np.zeros(len(cluster_id_ml_3))
    hii_reg_3_ml_3 = np.zeros(len(cluster_id_ml_3))
    hii_reg_5_ml_3 = np.zeros(len(cluster_id_ml_3))
    hii_reg_10_ml_3 = np.zeros(len(cluster_id_ml_3))

    if target == 'ngc0685':
        target_ext = 'ngc685'
    elif target == 'ngc0628c':
        target_ext = 'ngc628c'
    elif target == 'ngc0628e':
        target_ext = 'ngc628e'
    else:
        target_ext = target
    extra_file = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target_ext
    if os.path.isfile(extra_file):
        extra_tab_hdu = fits.open(extra_file)
        extra_data_table = extra_tab_hdu[1].data
        extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
        extra_h_alpha_intensity = extra_data_table['NBHa_intensity_medsub']
        extra_hii_reg_id = extra_data_table['NBHa_HIIreg']
        extra_hii_reg_id_3 = extra_data_table['NBHa_mask_medsub_lev3']
        extra_hii_reg_id_5 = extra_data_table['NBHa_mask_medsub_lev5']
        extra_hii_reg_id_10 = extra_data_table['NBHa_mask_medsub_lev10']

        for running_index, cluster_id in enumerate(cluster_id_hum_12):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_hum_12[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_reg_hum_12[running_index] = extra_hii_reg_id[index_extra_table]
            hii_reg_3_hum_12[running_index] = extra_hii_reg_id_3[index_extra_table]
            hii_reg_5_hum_12[running_index] = extra_hii_reg_id_5[index_extra_table]
            hii_reg_10_hum_12[running_index] = extra_hii_reg_id_10[index_extra_table]
        for running_index, cluster_id in enumerate(cluster_id_ml_12):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_ml_12[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_reg_ml_12[running_index] = extra_hii_reg_id[index_extra_table]
            hii_reg_3_ml_12[running_index] = extra_hii_reg_id_3[index_extra_table]
            hii_reg_5_ml_12[running_index] = extra_hii_reg_id_5[index_extra_table]
            hii_reg_10_ml_12[running_index] = extra_hii_reg_id_10[index_extra_table]
        for running_index, cluster_id in enumerate(cluster_id_hum_3):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_hum_3[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_reg_hum_3[running_index] = extra_hii_reg_id[index_extra_table]
            hii_reg_3_hum_3[running_index] = extra_hii_reg_id_3[index_extra_table]
            hii_reg_5_hum_3[running_index] = extra_hii_reg_id_5[index_extra_table]
            hii_reg_10_hum_3[running_index] = extra_hii_reg_id_10[index_extra_table]
        for running_index, cluster_id in enumerate(cluster_id_ml_3):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_ml_3[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_reg_ml_3[running_index] = extra_hii_reg_id[index_extra_table]
            hii_reg_3_ml_3[running_index] = extra_hii_reg_id_3[index_extra_table]
            hii_reg_5_ml_3[running_index] = extra_hii_reg_id_5[index_extra_table]
            hii_reg_10_ml_3[running_index] = extra_hii_reg_id_10[index_extra_table]
    else:
        print(extra_file, ' does not exist')

    hii_hum_12 = hii_reg_hum_12 > 0
    hii_ml_12 = hii_reg_ml_12 > 0
    hii_hum_3 = hii_reg_hum_3 > 0
    hii_ml_3 = hii_reg_ml_3 > 0

    h3_hum_12 = (h_alpha_intensity_hum_12 > 3) & (hii_reg_3_hum_12 > 0) & (age_hum_12 > 10)
    h3_ml_12 = (h_alpha_intensity_ml_12 > 3) & (hii_reg_3_ml_12 > 0) & (age_ml_12 > 10)
    h3_hum_3 = (h_alpha_intensity_hum_3 > 3) & (hii_reg_3_hum_3 > 0) & (age_hum_3 > 10)
    h3_ml_3 = (h_alpha_intensity_ml_3 > 3) & (hii_reg_3_ml_3 > 0) & (age_ml_3 > 10)

    h5_hum_12 = (h_alpha_intensity_hum_12 > 5) & (hii_reg_5_hum_12 > 0) & (age_hum_12 > 10)
    h5_ml_12 = (h_alpha_intensity_ml_12 > 5) & (hii_reg_5_ml_12 > 0) & (age_ml_12 > 10)
    h5_hum_3 = (h_alpha_intensity_hum_3 > 5) & (hii_reg_5_hum_3 > 0) & (age_hum_3 > 10)
    h5_ml_3 = (h_alpha_intensity_ml_3 > 5) & (hii_reg_5_ml_3 > 0) & (age_ml_3 > 10)

    h10_hum_12 = (h_alpha_intensity_hum_12 > 10) & (hii_reg_10_hum_12 > 0) & (age_hum_12 > 10)
    h10_ml_12 = (h_alpha_intensity_ml_12 > 10) & (hii_reg_10_ml_12 > 0) & (age_ml_12 > 10)
    h10_hum_3 = (h_alpha_intensity_hum_3 > 10) & (hii_reg_10_hum_3 > 0) & (age_hum_3 > 10)
    h10_ml_3 = (h_alpha_intensity_ml_3 > 10) & (hii_reg_10_ml_3 > 0) & (age_ml_3 > 10)

    print_str = target + ' & '

    print_str += str(sum(hii_hum_12) + sum(hii_hum_3)) + ' & '
    print_str += str(sum(hii_hum_12 * h3_hum_12) + sum(hii_hum_3 * h3_hum_3)) + ' & '
    print_str += str(sum(hii_hum_12 * h5_hum_12) + sum(hii_hum_3 * h5_hum_3)) + ' & '
    print_str += str(sum(hii_hum_12 * h10_hum_12) + sum(hii_hum_3 * h10_hum_3)) + ' & '

    print_str += str(sum(hii_ml_12) + sum(hii_ml_3)) + ' & '
    print_str += str(sum(hii_ml_12 * h3_ml_12) + sum(hii_ml_3 * h3_ml_3)) + ' & '
    print_str += str(sum(hii_ml_12 * h5_ml_12) + sum(hii_ml_3 * h5_ml_3)) + ' & '
    print_str += str(sum(hii_ml_12 * h10_ml_12) + sum(hii_ml_3 * h10_ml_3)) + ' \\\\'

    print(print_str)
