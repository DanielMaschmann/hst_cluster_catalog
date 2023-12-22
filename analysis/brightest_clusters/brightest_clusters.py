import os.path

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


catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')


catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

print(target_list)

for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_id_12_hum = catalog_access.get_hst_cc_phangs_id(target=target)
    hum_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    vgg_class_12_hum = catalog_access.get_hst_cc_class_ml_vgg(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    ebv_12_hum = catalog_access.get_hst_cc_ebv(target=target)
    mass_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
    ra_12_hum, dec_12_hum = catalog_access.get_hst_cc_coords_world(target=target)

    cluster_id_3_hum = catalog_access.get_hst_cc_phangs_id(target=target, cluster_class='class3')
    hum_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    vgg_class_3_hum = catalog_access.get_hst_cc_class_ml_vgg(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_3_hum = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    mass_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    ra_3_hum, dec_3_hum = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')

    cluster_id_hum = np.concatenate([cluster_id_12_hum, cluster_id_3_hum])
    hum_class_hum = np.concatenate([hum_class_12_hum, hum_class_3_hum])
    vgg_class_hum = np.concatenate([vgg_class_12_hum, vgg_class_3_hum])
    age_hum = np.concatenate([age_12_hum, age_3_hum])
    ebv_hum = np.concatenate([ebv_12_hum, ebv_3_hum])
    mass_hum = np.concatenate([mass_12_hum, mass_3_hum])
    ra_hum = np.concatenate([ra_12_hum, ra_3_hum])
    dec_hum = np.concatenate([dec_12_hum, dec_3_hum])



    cluster_id_12_ml = catalog_access.get_hst_cc_phangs_id(target=target, classify='ml')
    hum_class_12_ml = catalog_access.get_hst_cc_class_human(target=target, classify='ml')
    vgg_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_12_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    mass_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    ra_12_ml, dec_12_ml = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    x_12_ml, y_12_ml = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml')

    cluster_id_3_ml = catalog_access.get_hst_cc_phangs_id(target=target, classify='ml', cluster_class='class3')
    hum_class_3_ml = catalog_access.get_hst_cc_class_human(target=target, classify='ml', cluster_class='class3')
    vgg_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_3_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    mass_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
    ra_3_ml, dec_3_ml = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    x_3_ml, y_3_ml = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml', cluster_class='class3')

    cluster_id_ml = np.concatenate([cluster_id_12_ml, cluster_id_3_ml])
    hum_class_ml = np.concatenate([hum_class_12_ml, hum_class_3_ml])
    vgg_class_ml = np.concatenate([vgg_class_12_ml, vgg_class_3_ml])
    age_ml = np.concatenate([age_12_ml, age_3_ml])
    ebv_ml = np.concatenate([ebv_12_ml, ebv_3_ml])
    mass_ml = np.concatenate([mass_12_ml, mass_3_ml])
    ra_ml = np.concatenate([ra_12_ml, ra_3_ml])
    dec_ml = np.concatenate([dec_12_ml, dec_3_ml])
    x_ml = np.concatenate([x_12_ml, x_3_ml])
    y_ml = np.concatenate([y_12_ml, y_3_ml])

    mask_bright_clusters = (mass_ml > 1e6)
    print(sum(mask_bright_clusters))
    if sum(mask_bright_clusters) > 0:
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

        for ra, dec, x, y, cluster_id, hum_class, vgg_class, age, ebv, mass in zip(ra_ml[mask_bright_clusters],
                                                                      dec_ml[mask_bright_clusters],
                                                                             x_ml[mask_bright_clusters],
                                                                             y_ml[mask_bright_clusters],
                                                                      cluster_id_ml[mask_bright_clusters],
                                                                      hum_class_ml[mask_bright_clusters],
                                                                      vgg_class_ml[mask_bright_clusters],
                                                                      age_ml[mask_bright_clusters],
                                                                      ebv_ml[mask_bright_clusters],
                                                                            mass_ml[mask_bright_clusters]):

            str_line_1 = ('%s, '
                          'id_phangs_cluster = %i, '
                          'HUM_CLASS = %s, '
                          'ML_CLASS = %s, X=%.4f, Y=%.4f'
                          % (target, cluster_id, hum_class, vgg_class, x, y))
                          # add V-I color to the

            # indicate their id, mass, age, and name of parent galaxy in each stamp

            str_line_2 = (r'age= %i, E(B-V)=%.2f, log(M$_*$/M$_{\odot}$)=%.2f ' % (age, ebv, np.log10(mass)))

            try:
                fig = visualization_access.plot_multi_band_artifacts(ra=ra, dec=dec,
                                                                     str_line_1=str_line_1,
                                                                     str_line_2=str_line_2)

                if not os.path.isdir('plot_output/%s' % target):
                    os.makedirs('plot_output/%s' % target)
                fig.savefig('plot_output/%s/obj_in_%s_%i.png' % (target, target, cluster_id))
                plt.clf()
                plt.close("all")
            except ValueError:
                print('lalalal')
