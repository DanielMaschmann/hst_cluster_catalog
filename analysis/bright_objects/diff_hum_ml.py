import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

target_list = catalog_access.target_hst_cc

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


col_gal_name_hum = np.array([])
col_id_hum = np.array([])
col_ra_hum = np.array([])
col_dec_hum = np.array([])
col_class_hum = np.array([])

col_gal_name_ml = np.array([])
col_id_ml = np.array([])
col_ra_ml = np.array([])
col_dec_ml = np.array([])
col_class_ml = np.array([])

for target in target_list:

    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target

    dist = catalog_access.dist_dict[galaxy_name]['dist']

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    v_mag_hum_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    v_mag_hum_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', cluster_class='class3')
    abs_v_mag_hum_12 = hf.conv_mag2abs_mag(mag=v_mag_hum_12, dist=dist)
    abs_v_mag_hum_3 = hf.conv_mag2abs_mag(mag=v_mag_hum_3, dist=dist)
    ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
    ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
    cluster_id_hum_12 = catalog_access.get_hst_cc_phangs_id(target=target)
    cluster_id_hum_3 = catalog_access.get_hst_cc_phangs_id(target=target, cluster_class='class3')
    cluster_class_hum = np.concatenate([cluster_class_hum_12, cluster_class_hum_3])
    abs_v_mag_hum = np.concatenate([abs_v_mag_hum_12, abs_v_mag_hum_3])
    ra_hum = np.concatenate([ra_hum_12, ra_hum_3])
    dec_hum = np.concatenate([dec_hum_12, dec_hum_3])
    cluster_id_hum = np.concatenate([cluster_id_hum_12, cluster_id_hum_3])

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    v_mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band='F555W')
    v_mag_ml_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band='F555W', cluster_class='class3')
    abs_v_mag_ml_12 = hf.conv_mag2abs_mag(mag=v_mag_ml_12, dist=dist)
    abs_v_mag_ml_3 = hf.conv_mag2abs_mag(mag=v_mag_ml_3, dist=dist)
    ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    cluster_id_ml_12 = catalog_access.get_hst_cc_phangs_id(target=target, classify='ml')
    cluster_id_ml_3 = catalog_access.get_hst_cc_phangs_id(target=target, classify='ml', cluster_class='class3')
    cluster_class_ml = np.concatenate([cluster_class_ml_12, cluster_class_ml_3])
    abs_v_mag_ml = np.concatenate([abs_v_mag_ml_12, abs_v_mag_ml_3])
    ra_ml = np.concatenate([ra_ml_12, ra_ml_3])
    dec_ml = np.concatenate([dec_ml_12, dec_ml_3])
    cluster_id_ml = np.concatenate([cluster_id_ml_12, cluster_id_ml_3])


    for hum_clust_id_index, hum_clust_id in enumerate(cluster_id_hum):
        if (hum_clust_id not in cluster_id_ml) & (abs_v_mag_hum[hum_clust_id_index] < -12):
            print('target ', target, 'dist ', dist, ' ', hum_clust_id)
            col_gal_name_hum = np.concatenate([col_gal_name_hum, np.array([target])])
            col_id_hum = np.concatenate([col_id_hum, np.array([hum_clust_id])])
            col_ra_hum = np.concatenate([col_ra_hum, np.array([ra_hum[hum_clust_id_index]])])
            col_dec_hum = np.concatenate([col_dec_hum, np.array([dec_hum[hum_clust_id_index]])])
            col_class_hum = np.concatenate([col_class_hum, np.array([cluster_class_hum[hum_clust_id_index]])])

    for ml_clust_id_index, ml_clust_id in enumerate(cluster_id_ml):
        if (ml_clust_id not in cluster_id_hum) & (abs_v_mag_ml[ml_clust_id_index] < -12):
            # print('target ', target, 'dist ', dist, ' ', ml_clust_id)
            col_gal_name_ml = np.concatenate([col_gal_name_ml, np.array([target])])
            col_id_ml = np.concatenate([col_id_ml, np.array([ml_clust_id])])
            col_ra_ml = np.concatenate([col_ra_ml, np.array([ra_ml[ml_clust_id_index]])])
            col_dec_ml = np.concatenate([col_dec_ml, np.array([dec_ml[ml_clust_id_index]])])
            col_class_ml = np.concatenate([col_class_ml, np.array([cluster_class_ml[ml_clust_id_index]])])



from astropy.io import ascii

from astropy.table import Table

data_hum = Table()
data_hum['galaxy'] = col_gal_name_hum
data_hum['id'] = col_id_hum
data_hum['ra'] = col_ra_hum
data_hum['dec'] = col_dec_hum
data_hum['class'] = col_class_hum
print(data_hum)
ascii.write(data_hum, 'bright_sources_hum.csv', overwrite=True)


data_ml = Table()
data_ml['galaxy'] = col_gal_name_ml
data_ml['id'] = col_id_ml
data_ml['ra'] = col_ra_ml
data_ml['dec'] = col_dec_ml
data_ml['class'] = col_class_ml
print(len(data_ml))
ascii.write(data_ml, 'bright_sources_ml.csv', overwrite=True)



