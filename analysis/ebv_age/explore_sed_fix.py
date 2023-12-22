import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from cluster_cat_dr.visualization_tool import PhotVisualize

hst_data_path = '/media/benutzer/Sicherung/data/phangs_hst'
nircam_data_path = '/media/benutzer/Sicherung/data/phangs_jwst'
miri_data_path = '/media/benutzer/Sicherung/data/phangs_jwst'
hst_data_ver = 'v1.0'
nircam_data_ver = 'v0p9'
miri_data_ver = 'v0p9'

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path)

target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]


catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')



for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    index_12_ml = catalog_access.get_hst_cc_index(target=target, classify='ml')
    ml_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    hum_class_12_ml = catalog_access.get_hst_cc_class_human(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_12_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    ra_12_ml, dec_12_ml = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')

    index_3_ml = catalog_access.get_hst_cc_index(target=target, classify='ml', cluster_class='class3')
    ml_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    hum_class_3_ml = catalog_access.get_hst_cc_class_human(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_3_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    ra_3_ml, dec_3_ml = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')


    index_ml = np.concatenate([index_12_ml, index_3_ml])
    ml_class_ml = np.concatenate([ml_class_12_ml, ml_class_3_ml])
    hum_class_ml = np.concatenate([hum_class_12_ml, hum_class_3_ml])
    age_ml = np.concatenate([age_12_ml, age_3_ml])
    ebv_ml = np.concatenate([ebv_12_ml, ebv_3_ml])
    ra_ml = np.concatenate([ra_12_ml, ra_3_ml])
    dec_ml = np.concatenate([dec_12_ml, dec_3_ml])

    mask_problematic_values = (age_ml > 10) & (ebv_ml > 1.4)
    print(sum(mask_problematic_values))
    if sum(mask_problematic_values) > 0:
        if (target == 'ngc0628e') | (target == 'ngc0628c'):
            target_str = 'ngc0628'
        else:
            target_str = target
        visualization_access = PhotVisualize(
                                    target_name=target_str,
                                    hst_data_path=hst_data_path,
                                    nircam_data_path=nircam_data_path,
                                    miri_data_path=miri_data_path,
                                    hst_data_ver=hst_data_ver,
                                    nircam_data_ver=nircam_data_ver,
                                    miri_data_ver=miri_data_ver
                                )
        visualization_access.load_hst_nircam_miri_bands(flux_unit='MJy/sr', load_err=False)

        for ra, dec, cluster_id, ml_class, hum_class, age, ebv in zip(ra_ml[mask_problematic_values],
                                                                      dec_ml[mask_problematic_values],
                                                                      index_ml[mask_problematic_values],
                                                                      ml_class_ml[mask_problematic_values],
                                                                      hum_class_ml[mask_problematic_values],
                                                                      age_ml[mask_problematic_values],
                                                                      ebv_ml[mask_problematic_values]):

            str_line_1 = ('%s, '
                          'ID_PHANGS_CLUSTERS_v1p2 = %i, '
                          'HUM_CLASS = %s, '
                          'ML_CLASS = %s, '
                          % (target, cluster_id, hum_class, ml_class))
                          # add V-I color to the
            str_line_2 = ('age= %i, E(B-V)=%.2f ' % (age, ebv))

            fig = visualization_access.plot_multi_band_artifacts(ra=ra, dec=dec,
                                                                 str_line_1=str_line_1,
                                                                 str_line_2=str_line_2)


            fig.savefig('plot_output/problematic_sed_fit/obj_in_%s_%i.png' % (target, cluster_id))
            plt.clf()
            plt.close("all")