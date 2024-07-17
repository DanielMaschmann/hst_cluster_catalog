import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
from astropy.io import fits


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




target_name_hum = np.array([], dtype=str)
index_hum = np.array([])
phangs_cluster_id_hum = np.array([])
cluster_class_hum = np.array([])
color_color_class_hum = np.array([])

color_vi_hum_vega = np.array([])
color_ub_hum_vega = np.array([])
color_bv_hum_vega = np.array([])
color_nuvb_hum_vega = np.array([])

color_vi_err_hum_vega = np.array([])
color_ub_err_hum_vega = np.array([])
color_bv_err_hum_vega = np.array([])
color_nuvb_err_hum_vega = np.array([])

color_vi_hum_ab = np.array([])
color_ub_hum_ab = np.array([])
color_bv_hum_ab = np.array([])
color_nuvb_hum_ab = np.array([])

color_vi_err_hum_ab = np.array([])
color_ub_err_hum_ab = np.array([])
color_bv_err_hum_ab = np.array([])
color_nuvb_err_hum_ab = np.array([])

detect_nuv_hum = np.array([], dtype=bool)
detect_u_hum = np.array([], dtype=bool)
detect_b_hum = np.array([], dtype=bool)
detect_v_hum = np.array([], dtype=bool)
detect_i_hum = np.array([], dtype=bool)

age_hum = np.array([])
ebv_hum = np.array([])
mass_hum = np.array([])
ra_hum = np.array([])
dec_hum = np.array([])
x_hum = np.array([])
y_hum = np.array([])


for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name_str = 'ngc0628'
    else:
        galaxy_name_str = target
    inclination = catalog_access.get_target_incl(target=galaxy_name_str)
    print('target ', target, 'dist ', dist, 'inclination ', inclination)
    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'

    index_hum_12 = catalog_access.get_hst_cc_index(target=target)
    phangs_cluster_id_hum_12 = catalog_access.get_hst_cc_phangs_cluster_id(target=target)
    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_color_class_hum_12 = catalog_access.get_hst_color_color_class(target=target)

    color_vi_vega_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    color_ub_vega_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_bv_vega_hum_12 = catalog_access.get_hst_color_bv_vega(target=target)
    color_nuvb_vega_hum_12 = catalog_access.get_hst_color_nuvb_vega(target=target)

    color_vi_err_vega_hum_12 = catalog_access.get_hst_color_vi_err(target=target)
    color_ub_err_vega_hum_12 = catalog_access.get_hst_color_ub_err(target=target)
    color_bv_err_vega_hum_12 = catalog_access.get_hst_color_bv_err(target=target)
    color_nuvb_err_vega_hum_12 = catalog_access.get_hst_color_nuvb_err(target=target)

    color_vi_ab_hum_12 = catalog_access.get_hst_color_vi_ab(target=target)
    color_ub_ab_hum_12 = catalog_access.get_hst_color_ub_ab(target=target)
    color_bv_ab_hum_12 = catalog_access.get_hst_color_bv_ab(target=target)
    color_nuvb_ab_hum_12 = catalog_access.get_hst_color_nuvb_ab(target=target)

    color_vi_err_ab_hum_12 = catalog_access.get_hst_color_vi_err(target=target)
    color_ub_err_ab_hum_12 = catalog_access.get_hst_color_ub_err(target=target)
    color_bv_err_ab_hum_12 = catalog_access.get_hst_color_bv_err(target=target)
    color_nuvb_err_ab_hum_12 = catalog_access.get_hst_color_nuvb_err(target=target)

    detect_nuv_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F275W') > 0) &
                         (catalog_access.get_hst_cc_band_flux(target=target, band='F275W') >
                          3*catalog_access.get_hst_cc_band_flux_err(target=target, band='F275W')))
    detect_u_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F336W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, band='F336W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, band='F336W')))
    detect_b_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, band=b_band) >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, band=b_band)))
    detect_v_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, band='F555W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, band='F555W')))
    detect_i_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, band='F814W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, band='F814W')))

    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    ebv_hum_12 = catalog_access.get_hst_cc_ebv(target=target)
    mass_hum_12 = catalog_access.get_hst_cc_stellar_m(target=target)
    ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
    x_hum_12, y_hum_12 = catalog_access.get_hst_cc_coords_pix(target=target)



    index_hum_3 = catalog_access.get_hst_cc_index(target=target, cluster_class='class3')
    phangs_cluster_id_hum_3 = catalog_access.get_hst_cc_phangs_cluster_id(target=target, cluster_class='class3')
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_color_class_hum_3 = catalog_access.get_hst_color_color_class(target=target, cluster_class='class3')

    color_vi_vega_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    color_ub_vega_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_bv_vega_hum_3 = catalog_access.get_hst_color_bv_vega(target=target, cluster_class='class3')
    color_nuvb_vega_hum_3 = catalog_access.get_hst_color_nuvb_vega(target=target, cluster_class='class3')

    color_vi_err_vega_hum_3 = catalog_access.get_hst_color_vi_err(target=target, cluster_class='class3')
    color_ub_err_vega_hum_3 = catalog_access.get_hst_color_ub_err(target=target, cluster_class='class3')
    color_bv_err_vega_hum_3 = catalog_access.get_hst_color_bv_err(target=target, cluster_class='class3')
    color_nuvb_err_vega_hum_3 = catalog_access.get_hst_color_nuvb_err(target=target, cluster_class='class3')

    color_vi_ab_hum_3 = catalog_access.get_hst_color_vi_ab(target=target, cluster_class='class3')
    color_ub_ab_hum_3 = catalog_access.get_hst_color_ub_ab(target=target, cluster_class='class3')
    color_bv_ab_hum_3 = catalog_access.get_hst_color_bv_ab(target=target, cluster_class='class3')
    color_nuvb_ab_hum_3 = catalog_access.get_hst_color_nuvb_ab(target=target, cluster_class='class3')

    color_vi_err_ab_hum_3 = catalog_access.get_hst_color_vi_err(target=target, cluster_class='class3')
    color_ub_err_ab_hum_3 = catalog_access.get_hst_color_ub_err(target=target, cluster_class='class3')
    color_bv_err_ab_hum_3 = catalog_access.get_hst_color_bv_err(target=target, cluster_class='class3')
    color_nuvb_err_ab_hum_3 = catalog_access.get_hst_color_nuvb_err(target=target, cluster_class='class3')

    detect_nuv_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F275W') > 0) &
                         (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F275W') >
                          3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band='F275W')))
    detect_u_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F336W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F336W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band='F336W')))
    detect_b_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band=b_band)))
    detect_v_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band='F555W')))
    detect_i_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F814W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F814W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band='F814W')))

    age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_hum_3 = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    mass_hum_3 = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
    x_hum_3, y_hum_3 = catalog_access.get_hst_cc_coords_pix(target=target, cluster_class='class3')


    target_name_hum = np.concatenate([target_name_hum, np.array([target]*(len(index_hum_12) + len(index_hum_3)))])
    index_hum = np.concatenate([index_hum, index_hum_12, index_hum_3])
    phangs_cluster_id_hum = np.concatenate([phangs_cluster_id_hum, phangs_cluster_id_hum_12, phangs_cluster_id_hum_3])
    cluster_class_hum = np.concatenate([cluster_class_hum, cluster_class_hum_12, cluster_class_hum_3])
    color_color_class_hum = np.concatenate([color_color_class_hum, color_color_class_hum_12, color_color_class_hum_3])


    color_vi_hum_vega = np.concatenate([color_vi_hum_vega, color_vi_vega_hum_12, color_vi_vega_hum_3])
    color_ub_hum_vega = np.concatenate([color_ub_hum_vega, color_ub_vega_hum_12, color_ub_vega_hum_3])
    color_bv_hum_vega = np.concatenate([color_bv_hum_vega, color_bv_vega_hum_12, color_bv_vega_hum_3])
    color_nuvb_hum_vega = np.concatenate([color_nuvb_hum_vega, color_nuvb_vega_hum_12, color_nuvb_vega_hum_3])

    color_vi_err_hum_vega = np.concatenate([color_vi_err_hum_vega, color_vi_err_vega_hum_12, color_vi_err_vega_hum_3])
    color_ub_err_hum_vega = np.concatenate([color_ub_err_hum_vega, color_ub_err_vega_hum_12, color_ub_err_vega_hum_3])
    color_bv_err_hum_vega = np.concatenate([color_bv_err_hum_vega, color_bv_err_vega_hum_12, color_bv_err_vega_hum_3])
    color_nuvb_err_hum_vega = np.concatenate([color_nuvb_err_hum_vega, color_nuvb_err_vega_hum_12, color_nuvb_err_vega_hum_3])

    color_vi_hum_ab = np.concatenate([color_vi_hum_ab, color_vi_ab_hum_12, color_vi_ab_hum_3])
    color_ub_hum_ab = np.concatenate([color_ub_hum_ab, color_ub_ab_hum_12, color_ub_ab_hum_3])
    color_bv_hum_ab = np.concatenate([color_bv_hum_ab, color_bv_ab_hum_12, color_bv_ab_hum_3])
    color_nuvb_hum_ab = np.concatenate([color_nuvb_hum_ab, color_nuvb_ab_hum_12, color_nuvb_ab_hum_3])

    color_vi_err_hum_ab = np.concatenate([color_vi_err_hum_ab, color_vi_err_ab_hum_12, color_vi_err_ab_hum_3])
    color_ub_err_hum_ab = np.concatenate([color_ub_err_hum_ab, color_ub_err_ab_hum_12, color_ub_err_ab_hum_3])
    color_bv_err_hum_ab = np.concatenate([color_bv_err_hum_ab, color_bv_err_ab_hum_12, color_bv_err_ab_hum_3])
    color_nuvb_err_hum_ab = np.concatenate([color_nuvb_err_hum_ab, color_nuvb_err_ab_hum_12, color_nuvb_err_ab_hum_3])

    detect_nuv_hum = np.concatenate([detect_nuv_hum, detect_nuv_hum_12, detect_nuv_hum_3])
    detect_u_hum = np.concatenate([detect_u_hum, detect_u_hum_12, detect_u_hum_3])
    detect_b_hum = np.concatenate([detect_b_hum, detect_b_hum_12, detect_b_hum_3])
    detect_v_hum = np.concatenate([detect_v_hum, detect_v_hum_12, detect_v_hum_3])
    detect_i_hum = np.concatenate([detect_i_hum, detect_i_hum_12, detect_i_hum_3])

    age_hum = np.concatenate([age_hum, age_hum_12, age_hum_3])
    ebv_hum = np.concatenate([ebv_hum, ebv_hum_12, ebv_hum_3])
    mass_hum = np.concatenate([mass_hum, mass_hum_12, mass_hum_3])
    ra_hum = np.concatenate([ra_hum, ra_hum_12, ra_hum_3])
    dec_hum = np.concatenate([dec_hum, dec_hum_12, dec_hum_3])
    x_hum = np.concatenate([x_hum, x_hum_12, x_hum_3])
    y_hum = np.concatenate([y_hum, y_hum_12, y_hum_3])



np.save('data_output/target_name_hum.npy', target_name_hum)
np.save('data_output/index_hum.npy', index_hum)
np.save('data_output/phangs_cluster_id_hum.npy', phangs_cluster_id_hum)
np.save('data_output/cluster_class_hum.npy', cluster_class_hum)
np.save('data_output/color_color_class_hum.npy', color_color_class_hum)
np.save('data_output/color_vi_hum_vega.npy', color_vi_hum_vega)
np.save('data_output/color_ub_hum_vega.npy', color_ub_hum_vega)
np.save('data_output/color_bv_hum_vega.npy', color_bv_hum_vega)
np.save('data_output/color_nuvb_hum_vega.npy', color_nuvb_hum_vega)
np.save('data_output/color_vi_err_hum_vega.npy', color_vi_err_hum_vega)
np.save('data_output/color_ub_err_hum_vega.npy', color_ub_err_hum_vega)
np.save('data_output/color_bv_err_hum_vega.npy', color_bv_err_hum_vega)
np.save('data_output/color_nuvb_err_hum_vega.npy', color_nuvb_err_hum_vega)
np.save('data_output/color_vi_hum_ab.npy', color_vi_hum_ab)
np.save('data_output/color_ub_hum_ab.npy', color_ub_hum_ab)
np.save('data_output/color_bv_hum_ab.npy', color_bv_hum_ab)
np.save('data_output/color_nuvb_hum_ab.npy', color_nuvb_hum_ab)
np.save('data_output/color_vi_err_hum_ab.npy', color_vi_err_hum_ab)
np.save('data_output/color_ub_err_hum_ab.npy', color_ub_err_hum_ab)
np.save('data_output/color_bv_err_hum_ab.npy', color_bv_err_hum_ab)
np.save('data_output/color_nuvb_err_hum_ab.npy', color_nuvb_err_hum_ab)
np.save('data_output/detect_nuv_hum.npy', detect_nuv_hum)
np.save('data_output/detect_u_hum.npy', detect_u_hum)
np.save('data_output/detect_b_hum.npy', detect_b_hum)
np.save('data_output/detect_v_hum.npy', detect_v_hum)
np.save('data_output/detect_i_hum.npy', detect_i_hum)
np.save('data_output/age_hum.npy', age_hum)
np.save('data_output/ebv_hum.npy', ebv_hum)
np.save('data_output/mass_hum.npy', mass_hum)
np.save('data_output/ra_hum.npy', ra_hum)
np.save('data_output/dec_hum.npy', dec_hum)
np.save('data_output/x_hum.npy', x_hum)
np.save('data_output/y_hum.npy', y_hum)







