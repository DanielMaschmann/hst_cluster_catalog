""" bla bla bla """
import numpy as np
import matplotlib.pyplot as plt
import photometry_tools


# get access to HST cluster catalog
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

cc_target_list = catalog_access.target_hst_cc
# target_list = catalog_access.phangs_galaxy_list

dist_list = []
for target in cc_target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
# sort target list after distance
sort = np.argsort(dist_list)
cc_target_list = np.array(cc_target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=cc_target_list)
catalog_access.load_hst_cc_list(target_list=cc_target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=cc_target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=cc_target_list, classify='ml', cluster_class='class3')

array_f555w_abs_mag_hum_12 = np.array([])
max_f555w_abs_mag_hum_12 = np.zeros(len(cc_target_list))
min_f555w_abs_mag_hum_12 = np.zeros(len(cc_target_list))
mean_f555w_abs_mag_hum_12 = np.zeros(len(cc_target_list))
median_f555w_abs_mag_hum_12 = np.zeros(len(cc_target_list))
p16_f555w_abs_mag_hum_12 = np.zeros(len(cc_target_list))
p84_f555w_abs_mag_hum_12 = np.zeros(len(cc_target_list))

array_f555w_abs_mag_hum_1 = np.array([])
max_f555w_abs_mag_hum_1 = np.zeros(len(cc_target_list))
min_f555w_abs_mag_hum_1 = np.zeros(len(cc_target_list))
mean_f555w_abs_mag_hum_1 = np.zeros(len(cc_target_list))
median_f555w_abs_mag_hum_1 = np.zeros(len(cc_target_list))
p16_f555w_abs_mag_hum_1 = np.zeros(len(cc_target_list))
p84_f555w_abs_mag_hum_1 = np.zeros(len(cc_target_list))

array_f555w_abs_mag_hum_2 = np.array([])
max_f555w_abs_mag_hum_2 = np.zeros(len(cc_target_list))
min_f555w_abs_mag_hum_2 = np.zeros(len(cc_target_list))
mean_f555w_abs_mag_hum_2 = np.zeros(len(cc_target_list))
median_f555w_abs_mag_hum_2 = np.zeros(len(cc_target_list))
p16_f555w_abs_mag_hum_2 = np.zeros(len(cc_target_list))
p84_f555w_abs_mag_hum_2 = np.zeros(len(cc_target_list))

array_f555w_abs_mag_ml_12 = np.array([])
max_f555w_abs_mag_ml_12 = np.zeros(len(cc_target_list))
min_f555w_abs_mag_ml_12 = np.zeros(len(cc_target_list))
mean_f555w_abs_mag_ml_12 = np.zeros(len(cc_target_list))
median_f555w_abs_mag_ml_12 = np.zeros(len(cc_target_list))
p16_f555w_abs_mag_ml_12 = np.zeros(len(cc_target_list))
p84_f555w_abs_mag_ml_12 = np.zeros(len(cc_target_list))

array_f555w_abs_mag_ml_1 = np.array([])
max_f555w_abs_mag_ml_1 = np.zeros(len(cc_target_list))
min_f555w_abs_mag_ml_1 = np.zeros(len(cc_target_list))
mean_f555w_abs_mag_ml_1 = np.zeros(len(cc_target_list))
median_f555w_abs_mag_ml_1 = np.zeros(len(cc_target_list))
p16_f555w_abs_mag_ml_1 = np.zeros(len(cc_target_list))
p84_f555w_abs_mag_ml_1 = np.zeros(len(cc_target_list))

array_f555w_abs_mag_ml_2 = np.array([])
max_f555w_abs_mag_ml_2 = np.zeros(len(cc_target_list))
min_f555w_abs_mag_ml_2 = np.zeros(len(cc_target_list))
mean_f555w_abs_mag_ml_2 = np.zeros(len(cc_target_list))
median_f555w_abs_mag_ml_2 = np.zeros(len(cc_target_list))
p16_f555w_abs_mag_ml_2 = np.zeros(len(cc_target_list))
p84_f555w_abs_mag_ml_2 = np.zeros(len(cc_target_list))

array_mstar_hum_12 = np.array([])
p99_mstar_hum_12 = np.zeros(len(cc_target_list))
p1_mstar_hum_12 = np.zeros(len(cc_target_list))
max_mstar_hum_12 = np.zeros(len(cc_target_list))
min_mstar_hum_12 = np.zeros(len(cc_target_list))
mean_mstar_hum_12 = np.zeros(len(cc_target_list))
median_mstar_hum_12 = np.zeros(len(cc_target_list))
p16_mstar_hum_12 = np.zeros(len(cc_target_list))
p84_mstar_hum_12 = np.zeros(len(cc_target_list))

array_mstar_hum_1 = np.array([])
p99_mstar_hum_1 = np.zeros(len(cc_target_list))
p1_mstar_hum_1 = np.zeros(len(cc_target_list))
max_mstar_hum_1 = np.zeros(len(cc_target_list))
min_mstar_hum_1 = np.zeros(len(cc_target_list))
mean_mstar_hum_1 = np.zeros(len(cc_target_list))
median_mstar_hum_1 = np.zeros(len(cc_target_list))
p16_mstar_hum_1 = np.zeros(len(cc_target_list))
p84_mstar_hum_1 = np.zeros(len(cc_target_list))

array_mstar_hum_2 = np.array([])
p99_mstar_hum_2 = np.zeros(len(cc_target_list))
p1_mstar_hum_2 = np.zeros(len(cc_target_list))
max_mstar_hum_2 = np.zeros(len(cc_target_list))
min_mstar_hum_2 = np.zeros(len(cc_target_list))
mean_mstar_hum_2 = np.zeros(len(cc_target_list))
median_mstar_hum_2 = np.zeros(len(cc_target_list))
p16_mstar_hum_2 = np.zeros(len(cc_target_list))
p84_mstar_hum_2 = np.zeros(len(cc_target_list))

array_mstar_ml_12 = np.array([])
p99_mstar_ml_12 = np.zeros(len(cc_target_list))
p1_mstar_ml_12 = np.zeros(len(cc_target_list))
max_mstar_ml_12 = np.zeros(len(cc_target_list))
min_mstar_ml_12 = np.zeros(len(cc_target_list))
mean_mstar_ml_12 = np.zeros(len(cc_target_list))
median_mstar_ml_12 = np.zeros(len(cc_target_list))
p16_mstar_ml_12 = np.zeros(len(cc_target_list))
p84_mstar_ml_12 = np.zeros(len(cc_target_list))

array_mstar_ml_1 = np.array([])
p99_mstar_ml_1 = np.zeros(len(cc_target_list))
p1_mstar_ml_1 = np.zeros(len(cc_target_list))
max_mstar_ml_1 = np.zeros(len(cc_target_list))
min_mstar_ml_1 = np.zeros(len(cc_target_list))
mean_mstar_ml_1 = np.zeros(len(cc_target_list))
median_mstar_ml_1 = np.zeros(len(cc_target_list))
p16_mstar_ml_1 = np.zeros(len(cc_target_list))
p84_mstar_ml_1 = np.zeros(len(cc_target_list))


array_mstar_ml_2 = np.array([])
p99_mstar_ml_2 = np.zeros(len(cc_target_list))
p1_mstar_ml_2 = np.zeros(len(cc_target_list))
max_mstar_ml_2 = np.zeros(len(cc_target_list))
min_mstar_ml_2 = np.zeros(len(cc_target_list))
mean_mstar_ml_2 = np.zeros(len(cc_target_list))
median_mstar_ml_2 = np.zeros(len(cc_target_list))
p16_mstar_ml_2 = np.zeros(len(cc_target_list))
p84_mstar_ml_2 = np.zeros(len(cc_target_list))

mask_deep = np.zeros(len(cc_target_list), dtype=bool)
array_dist_hum_12 = np.array([])
array_dist_hum_1 = np.array([])
array_dist_hum_2 = np.array([])
array_dist_ml_12 = np.array([])
array_dist_ml_1 = np.array([])
array_dist_ml_2 = np.array([])

for index, target in enumerate(cc_target_list):
    print(cc_target_list[index])
    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    mask_class_hum_1 = cluster_class_hum_12 == 1
    mask_class_hum_2 = cluster_class_hum_12 == 2
    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    mask_class_ml_1 = cluster_class_ml_12 == 1
    mask_class_ml_2 = cluster_class_ml_12 == 2

    f555w_mag_hum_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    f555w_mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band='F555W')

    mstar_hum_12 = catalog_access.get_hst_cc_stellar_m(target=target)
    mstar_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')

    f555w_abs_mag_hum_12 = f555w_mag_hum_12 - 5*np.log10(dist_list[index] * 1e6) + 5
    f555w_abs_mag_ml_12 = f555w_mag_ml_12 - 5*np.log10(dist_list[index] * 1e6) + 5

    array_f555w_abs_mag_hum_12 = np.concatenate([array_f555w_abs_mag_hum_12, f555w_abs_mag_hum_12])
    array_f555w_abs_mag_hum_1 = np.concatenate([array_f555w_abs_mag_hum_1, f555w_abs_mag_hum_12[mask_class_hum_1]])
    array_f555w_abs_mag_hum_2 = np.concatenate([array_f555w_abs_mag_hum_2, f555w_abs_mag_hum_12[mask_class_hum_2]])

    array_f555w_abs_mag_ml_12 = np.concatenate([array_f555w_abs_mag_ml_12, f555w_abs_mag_ml_12])
    array_f555w_abs_mag_ml_1 = np.concatenate([array_f555w_abs_mag_ml_1, f555w_abs_mag_ml_12[mask_class_ml_1]])
    array_f555w_abs_mag_ml_2 = np.concatenate([array_f555w_abs_mag_ml_2, f555w_abs_mag_ml_12[mask_class_ml_2]])

    array_mstar_hum_12 = np.concatenate([array_mstar_hum_12, mstar_hum_12])
    array_mstar_hum_1 = np.concatenate([array_mstar_hum_1, mstar_hum_12[mask_class_hum_1]])
    array_mstar_hum_2 = np.concatenate([array_mstar_hum_2, mstar_hum_12[mask_class_hum_2]])
    array_mstar_ml_12 = np.concatenate([array_mstar_ml_12, mstar_ml_12])
    array_mstar_ml_1 = np.concatenate([array_mstar_ml_1, mstar_ml_12[mask_class_ml_1]])
    array_mstar_ml_2 = np.concatenate([array_mstar_ml_2, mstar_ml_12[mask_class_ml_2]])









    array_dist_hum_12 = np.concatenate([array_dist_hum_12, np.ones(len(f555w_abs_mag_hum_12)) * dist_list[index]])
    array_dist_hum_1 = np.concatenate([array_dist_hum_1, np.ones(len(f555w_abs_mag_hum_12[mask_class_hum_1])) * dist_list[index]])
    array_dist_hum_2 = np.concatenate([array_dist_hum_2, np.ones(len(f555w_abs_mag_hum_12[mask_class_hum_2])) * dist_list[index]])
    array_dist_ml_12 = np.concatenate([array_dist_ml_12, np.ones(len(f555w_abs_mag_ml_12)) * dist_list[index]])
    array_dist_ml_1 = np.concatenate([array_dist_ml_1, np.ones(len(f555w_abs_mag_ml_12[mask_class_ml_1])) * dist_list[index]])
    array_dist_ml_2 = np.concatenate([array_dist_ml_2, np.ones(len(f555w_abs_mag_ml_12[mask_class_ml_2])) * dist_list[index]])

    median_f555w_mag_ml_12 = np.nanmedian(f555w_mag_ml_12[(f555w_mag_ml_12 > 18) & (f555w_mag_ml_12 < 27)])
    faintest_f555w_band_mag_hum_12 = np.nanmax(f555w_mag_hum_12[(f555w_mag_hum_12 > 18) & (f555w_mag_hum_12 < 27)])
    if median_f555w_mag_ml_12 > faintest_f555w_band_mag_hum_12:
        mask_deep[index] = False
    else:
        mask_deep[index] = True


    max_f555w_abs_mag_hum_12[index] = np.nanmax(f555w_abs_mag_hum_12)
    min_f555w_abs_mag_hum_12[index] = np.nanmin(f555w_abs_mag_hum_12)
    mean_f555w_abs_mag_hum_12[index] = np.nanmean(f555w_abs_mag_hum_12)
    median_f555w_abs_mag_hum_12[index] = np.nanmedian(f555w_abs_mag_hum_12)
    p16_f555w_abs_mag_hum_12[index] = np.percentile(f555w_abs_mag_hum_12, 16)
    p84_f555w_abs_mag_hum_12[index] = np.percentile(f555w_abs_mag_hum_12, 84)

    max_f555w_abs_mag_hum_1[index] = np.nanmax(f555w_abs_mag_hum_12[mask_class_hum_1])
    min_f555w_abs_mag_hum_1[index] = np.nanmin(f555w_abs_mag_hum_12[mask_class_hum_1])
    mean_f555w_abs_mag_hum_1[index] = np.nanmean(f555w_abs_mag_hum_12[mask_class_hum_1])
    median_f555w_abs_mag_hum_1[index] = np.nanmedian(f555w_abs_mag_hum_12[mask_class_hum_1])
    p16_f555w_abs_mag_hum_1[index] = np.percentile(f555w_abs_mag_hum_12[mask_class_hum_1], 16)
    p84_f555w_abs_mag_hum_1[index] = np.percentile(f555w_abs_mag_hum_12[mask_class_hum_1], 84)

    max_f555w_abs_mag_hum_2[index] = np.nanmax(f555w_abs_mag_hum_12[mask_class_hum_2])
    min_f555w_abs_mag_hum_2[index] = np.nanmin(f555w_abs_mag_hum_12[mask_class_hum_2])
    mean_f555w_abs_mag_hum_2[index] = np.nanmean(f555w_abs_mag_hum_12[mask_class_hum_2])
    median_f555w_abs_mag_hum_2[index] = np.nanmedian(f555w_abs_mag_hum_12[mask_class_hum_2])
    p16_f555w_abs_mag_hum_2[index] = np.percentile(f555w_abs_mag_hum_12[mask_class_hum_2], 16)
    p84_f555w_abs_mag_hum_2[index] = np.percentile(f555w_abs_mag_hum_12[mask_class_hum_2], 84)

    max_f555w_abs_mag_ml_12[index] = np.nanmax(f555w_abs_mag_ml_12)
    min_f555w_abs_mag_ml_12[index] = np.nanmin(f555w_abs_mag_ml_12)
    mean_f555w_abs_mag_ml_12[index] = np.nanmean(f555w_abs_mag_ml_12)
    median_f555w_abs_mag_ml_12[index] = np.nanmedian(f555w_abs_mag_ml_12)
    p16_f555w_abs_mag_ml_12[index] = np.percentile(f555w_abs_mag_ml_12, 16)
    p84_f555w_abs_mag_ml_12[index] = np.percentile(f555w_abs_mag_ml_12, 84)

    max_f555w_abs_mag_ml_1[index] = np.nanmax(f555w_abs_mag_ml_12[mask_class_ml_1])
    min_f555w_abs_mag_ml_1[index] = np.nanmin(f555w_abs_mag_ml_12[mask_class_ml_1])
    mean_f555w_abs_mag_ml_1[index] = np.nanmean(f555w_abs_mag_ml_12[mask_class_ml_1])
    median_f555w_abs_mag_ml_1[index] = np.nanmedian(f555w_abs_mag_ml_12[mask_class_ml_1])
    p16_f555w_abs_mag_ml_1[index] = np.percentile(f555w_abs_mag_ml_12[mask_class_ml_1], 16)
    p84_f555w_abs_mag_ml_1[index] = np.percentile(f555w_abs_mag_ml_12[mask_class_ml_1], 84)

    max_f555w_abs_mag_ml_2[index] = np.nanmax(f555w_abs_mag_ml_12[mask_class_ml_2])
    min_f555w_abs_mag_ml_2[index] = np.nanmin(f555w_abs_mag_ml_12[mask_class_ml_2])
    mean_f555w_abs_mag_ml_2[index] = np.nanmean(f555w_abs_mag_ml_12[mask_class_ml_2])
    median_f555w_abs_mag_ml_2[index] = np.nanmedian(f555w_abs_mag_ml_12[mask_class_ml_2])
    p16_f555w_abs_mag_ml_2[index] = np.percentile(f555w_abs_mag_ml_12[mask_class_ml_2], 16)
    p84_f555w_abs_mag_ml_2[index] = np.percentile(f555w_abs_mag_ml_12[mask_class_ml_2], 84)


    mask_good_m_star_hum_12 = mstar_hum_12 > 0

    p99_mstar_hum_12[index] = np.percentile(mstar_hum_12[mask_good_m_star_hum_12], 99)
    p1_mstar_hum_12[index] = np.percentile(mstar_hum_12[mask_good_m_star_hum_12], 1)
    max_mstar_hum_12[index] = np.nanmax(mstar_hum_12[mask_good_m_star_hum_12])
    min_mstar_hum_12[index] = np.nanmin(mstar_hum_12[mask_good_m_star_hum_12])
    mean_mstar_hum_12[index] = np.nanmean(mstar_hum_12[mask_good_m_star_hum_12])
    median_mstar_hum_12[index] = np.nanmedian(mstar_hum_12[mask_good_m_star_hum_12])
    p16_mstar_hum_12[index] = np.percentile(mstar_hum_12[mask_good_m_star_hum_12], 16)
    p84_mstar_hum_12[index] = np.percentile(mstar_hum_12[mask_good_m_star_hum_12], 84)

    p99_mstar_hum_1[index] = np.percentile(mstar_hum_12[mask_class_hum_1 * mask_good_m_star_hum_12], 99)
    p1_mstar_hum_1[index] = np.percentile(mstar_hum_12[mask_class_hum_1 * mask_good_m_star_hum_12], 1)
    max_mstar_hum_1[index] = np.nanmax(mstar_hum_12[mask_class_hum_1 * mask_good_m_star_hum_12])
    min_mstar_hum_1[index] = np.nanmin(mstar_hum_12[mask_class_hum_1 * mask_good_m_star_hum_12])
    mean_mstar_hum_1[index] = np.nanmean(mstar_hum_12[mask_class_hum_1 * mask_good_m_star_hum_12])
    median_mstar_hum_1[index] = np.nanmedian(mstar_hum_12[mask_class_hum_1 * mask_good_m_star_hum_12])
    p16_mstar_hum_1[index] = np.percentile(mstar_hum_12[mask_class_hum_1 * mask_good_m_star_hum_12], 16)
    p84_mstar_hum_1[index] = np.percentile(mstar_hum_12[mask_class_hum_1 * mask_good_m_star_hum_12], 84)

    p99_mstar_hum_2[index] = np.percentile(mstar_hum_12[mask_class_hum_2 * mask_good_m_star_hum_12], 99)
    p1_mstar_hum_2[index] = np.percentile(mstar_hum_12[mask_class_hum_2 * mask_good_m_star_hum_12], 1)
    max_mstar_hum_2[index] = np.nanmax(mstar_hum_12[mask_class_hum_2 * mask_good_m_star_hum_12])
    min_mstar_hum_2[index] = np.nanmin(mstar_hum_12[mask_class_hum_2 * mask_good_m_star_hum_12])
    mean_mstar_hum_2[index] = np.nanmean(mstar_hum_12[mask_class_hum_2 * mask_good_m_star_hum_12])
    median_mstar_hum_2[index] = np.nanmedian(mstar_hum_12[mask_class_hum_2 * mask_good_m_star_hum_12])
    p16_mstar_hum_2[index] = np.percentile(mstar_hum_12[mask_class_hum_2 * mask_good_m_star_hum_12], 16)
    p84_mstar_hum_2[index] = np.percentile(mstar_hum_12[mask_class_hum_2 * mask_good_m_star_hum_12], 84)


    mask_good_m_star_ml_12 = mstar_ml_12 > 0

    p99_mstar_ml_12[index] = np.percentile(mstar_ml_12[mask_good_m_star_ml_12], 99)
    p1_mstar_ml_12[index] = np.percentile(mstar_ml_12[mask_good_m_star_ml_12], 1)
    max_mstar_ml_12[index] = np.nanmax(mstar_ml_12[mask_good_m_star_ml_12])
    min_mstar_ml_12[index] = np.nanmin(mstar_ml_12[mask_good_m_star_ml_12])
    mean_mstar_ml_12[index] = np.nanmean(mstar_ml_12[mask_good_m_star_ml_12])
    median_mstar_ml_12[index] = np.nanmedian(mstar_ml_12[mask_good_m_star_ml_12])
    p16_mstar_ml_12[index] = np.percentile(mstar_ml_12[mask_good_m_star_ml_12], 16)
    p84_mstar_ml_12[index] = np.percentile(mstar_ml_12[mask_good_m_star_ml_12], 84)

    p99_mstar_ml_1[index] = np.percentile(mstar_ml_12[mask_class_ml_1 * mask_good_m_star_ml_12], 99)
    p1_mstar_ml_1[index] = np.percentile(mstar_ml_12[mask_class_ml_1 * mask_good_m_star_ml_12], 1)
    max_mstar_ml_1[index] = np.nanmax(mstar_ml_12[mask_class_ml_1 * mask_good_m_star_ml_12])
    min_mstar_ml_1[index] = np.nanmin(mstar_ml_12[mask_class_ml_1 * mask_good_m_star_ml_12])
    mean_mstar_ml_1[index] = np.nanmean(mstar_ml_12[mask_class_ml_1 * mask_good_m_star_ml_12])
    median_mstar_ml_1[index] = np.nanmedian(mstar_ml_12[mask_class_ml_1 * mask_good_m_star_ml_12])
    p16_mstar_ml_1[index] = np.percentile(mstar_ml_12[mask_class_ml_1 * mask_good_m_star_ml_12], 16)
    p84_mstar_ml_1[index] = np.percentile(mstar_ml_12[mask_class_ml_1 * mask_good_m_star_ml_12], 84)

    p99_mstar_ml_2[index] = np.percentile(mstar_ml_12[mask_class_ml_2 * mask_good_m_star_ml_12], 99)
    p1_mstar_ml_2[index] = np.percentile(mstar_ml_12[mask_class_ml_2 * mask_good_m_star_ml_12], 1)
    max_mstar_ml_2[index] = np.nanmax(mstar_ml_12[mask_class_ml_2 * mask_good_m_star_ml_12])
    min_mstar_ml_2[index] = np.nanmin(mstar_ml_12[mask_class_ml_2 * mask_good_m_star_ml_12])
    mean_mstar_ml_2[index] = np.nanmean(mstar_ml_12[mask_class_ml_2 * mask_good_m_star_ml_12])
    median_mstar_ml_2[index] = np.nanmedian(mstar_ml_12[mask_class_ml_2 * mask_good_m_star_ml_12])
    p16_mstar_ml_2[index] = np.percentile(mstar_ml_12[mask_class_ml_2 * mask_good_m_star_ml_12], 16)
    p84_mstar_ml_2[index] = np.percentile(mstar_ml_12[mask_class_ml_2 * mask_good_m_star_ml_12], 84)

# print some statistics
mask_close_hum = array_dist_hum_12 < 14
mask_far_hum = array_dist_hum_12 > 14
mask_close_ml = array_dist_ml_12 < 14
mask_far_ml = array_dist_ml_12 > 14

mask_good_masses_hum = array_mstar_hum_12 > 0
mask_good_masses_ml = array_mstar_ml_12 > 0


print('below 14 Mpc Hum 12 Mean V-mag', np.nanmean(array_f555w_abs_mag_hum_12[mask_close_hum]))
print('above 14 Mpc Hum 12 Mean V-mag', np.nanmean(array_f555w_abs_mag_hum_12[mask_far_hum]))

print('below 14 Mpc ML 12 Mean V-mag', np.nanmean(array_f555w_abs_mag_ml_12[mask_close_ml]))
print('above 14 Mpc ML 12 Mean V-mag', np.nanmean(array_f555w_abs_mag_ml_12[mask_far_ml]))

print('below 14 Mpc Hum 12 Mean Mstar', np.nanmean(array_mstar_hum_12[mask_close_hum * mask_good_masses_hum]))
print('above 14 Mpc Hum 12 Mean Mstar', np.nanmean(array_mstar_hum_12[mask_far_hum * mask_good_masses_hum]))

print('below 14 Mpc ML 12 Mean Mstar', np.nanmean(array_mstar_ml_12[mask_close_ml * mask_good_masses_ml]))
print('above 14 Mpc ML 12 Mean Mstar', np.nanmean(array_mstar_ml_12[mask_far_ml * mask_good_masses_ml]))

print('--------------------------------')

print('below 14 Mpc Hum 12 Median V-mag', np.nanmedian(array_f555w_abs_mag_hum_12[mask_close_hum]))
print('above 14 Mpc Hum 12 Median V-mag', np.nanmedian(array_f555w_abs_mag_hum_12[mask_far_hum]))

print('below 14 Mpc ML 12 Median V-mag', np.nanmedian(array_f555w_abs_mag_ml_12[mask_close_ml]))
print('above 14 Mpc ML 12 Median V-mag', np.nanmedian(array_f555w_abs_mag_ml_12[mask_far_ml]))

print('below 14 Mpc Hum 12 Median Mstar', np.log10(np.nanmedian(array_mstar_hum_12[mask_close_hum * mask_good_masses_hum])))
print('above 14 Mpc Hum 12 Median Mstar', np.log10(np.nanmedian(array_mstar_hum_12[mask_far_hum * mask_good_masses_hum])))

print('below 14 Mpc ML 12 Median Mstar', np.log10(np.nanmedian(array_mstar_ml_12[mask_close_ml * mask_good_masses_ml])))
print('above 14 Mpc ML 12 Median Mstar', np.log10(np.nanmedian(array_mstar_ml_12[mask_far_ml * mask_good_masses_ml])))

print('--------------------------------')


fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, sharey='row', figsize=(28, 28))
fontsize = 30

scatter_dot_size = 140
scatter_star_size = 140

err_bar_dot_size = 12
err_bar_star_size = 12


for index in range(len(dist_list)):
    if not mask_deep[index]:
        ax[0, 0].scatter(dist_list[index], min_f555w_abs_mag_hum_1[index], s=scatter_star_size, linewidth=2,  marker='o',
                         facecolor='None', color='tab:blue')
        ax[0, 0].errorbar(dist_list[index], median_f555w_abs_mag_hum_1[index],
                          yerr=np.array([[median_f555w_abs_mag_hum_1[index] - p16_f555w_abs_mag_hum_1[index]],
                                         [p84_f555w_abs_mag_hum_1[index] - median_f555w_abs_mag_hum_1[index]]]),
                          fmt='o', mfc='w', ms=err_bar_star_size, color='darkslategrey')
        ax[0, 0].scatter(dist_list[index], max_f555w_abs_mag_hum_1[index], s=scatter_star_size, linewidth=2,  marker='o',
                         facecolor='None', color='tab:red')

        ax[1, 0].scatter(dist_list[index], min_f555w_abs_mag_hum_2[index], s=scatter_star_size, linewidth=2,  marker='o',
                         facecolor='None', color='tab:blue')
        ax[1, 0].errorbar(dist_list[index], median_f555w_abs_mag_hum_2[index],
                          yerr=np.array([[median_f555w_abs_mag_hum_2[index] - p16_f555w_abs_mag_hum_2[index]],
                                         [p84_f555w_abs_mag_hum_2[index] - median_f555w_abs_mag_hum_2[index]]]),
                          fmt='o', mfc='w', ms=err_bar_star_size, color='darkslategrey')
        ax[1, 0].scatter(dist_list[index], max_f555w_abs_mag_hum_2[index], s=scatter_star_size, linewidth=2,  marker='o',
                         facecolor='None', color='tab:red')

    else:
        ax[0, 0].scatter(dist_list[index], min_f555w_abs_mag_hum_1[index], s=scatter_dot_size, color='tab:blue')
        ax[0, 0].errorbar(dist_list[index], median_f555w_abs_mag_hum_1[index],
                          yerr=np.array([[median_f555w_abs_mag_hum_1[index] - p16_f555w_abs_mag_hum_1[index]],
                                          [p84_f555w_abs_mag_hum_1[index] - median_f555w_abs_mag_hum_1[index]]]),
                          fmt='o', ms=err_bar_dot_size, color='darkslategrey')
        ax[0, 0].scatter(dist_list[index], max_f555w_abs_mag_hum_1[index], s=scatter_dot_size, color='tab:red')

        ax[1, 0].scatter(dist_list[index], min_f555w_abs_mag_hum_2[index], s=scatter_dot_size, color='tab:blue')
        ax[1, 0].errorbar(dist_list[index], median_f555w_abs_mag_hum_2[index],
                          yerr=np.array([[median_f555w_abs_mag_hum_2[index] - p16_f555w_abs_mag_hum_2[index]],
                                          [p84_f555w_abs_mag_hum_2[index] - median_f555w_abs_mag_hum_2[index]]]),
                          fmt='o', ms=err_bar_dot_size, color='darkslategrey')
        ax[1, 0].scatter(dist_list[index], max_f555w_abs_mag_hum_2[index], s=scatter_dot_size, color='tab:red')


ax[0, 0].scatter([], [], s=scatter_dot_size, color='tab:blue', label='Brightest cluster')
# ax[0, 0].errorbar([], [], yerr=[], fmt='o', ms=err_bar_dot_size, color='darkslategrey', label='Median value')
ax[0, 0].scatter([], [], s=scatter_dot_size, color='darkslategrey', label='Median value')
ax[0, 0].scatter([], [], s=scatter_dot_size, color='tab:red', label='Faintest cluster')


ax[0, 1].scatter(dist_list, min_f555w_abs_mag_ml_1, s=120, color='tab:blue', label='Min value')
ax[0, 1].errorbar(dist_list, median_f555w_abs_mag_ml_1,
                  yerr=[median_f555w_abs_mag_ml_1 - p16_f555w_abs_mag_ml_1,
                        p84_f555w_abs_mag_ml_1 - median_f555w_abs_mag_ml_1],
                  fmt='o', ms=err_bar_dot_size, color='darkslategrey', label='Median value')
ax[0, 1].scatter(dist_list, max_f555w_abs_mag_ml_1, s=120, color='tab:red', label='Max value')

ax[1, 1].scatter(dist_list, min_f555w_abs_mag_ml_2, s=120, color='tab:blue', label='Min value')
ax[1, 1].errorbar(dist_list, median_f555w_abs_mag_ml_2,
                  yerr=[median_f555w_abs_mag_ml_2 - p16_f555w_abs_mag_ml_2,
                        p84_f555w_abs_mag_ml_2 - median_f555w_abs_mag_ml_2],
                  fmt='o', ms=err_bar_dot_size, color='darkslategrey', label='Median value')
ax[1, 1].scatter(dist_list, max_f555w_abs_mag_ml_2, s=120, color='tab:red', label='Max value')



ax[2, 0].scatter(dist_list, max_mstar_hum_1, s=scatter_dot_size, color='tab:purple')
ax[2, 0].errorbar(dist_list, median_mstar_hum_1,
                  yerr=[median_mstar_hum_1 - p16_mstar_hum_1,
                        p84_mstar_hum_1 - median_mstar_hum_1],
                  fmt='o', ms=err_bar_dot_size, color='darkslategrey')
ax[2, 0].scatter(dist_list, min_mstar_hum_1, s=scatter_dot_size, color='tab:green')

ax[3, 0].scatter(dist_list, max_mstar_hum_2, s=scatter_dot_size, color='tab:purple')
ax[3, 0].errorbar(dist_list, median_mstar_hum_2,
                  yerr=[median_mstar_hum_2 - p16_mstar_hum_2,
                        p84_mstar_hum_2 - median_mstar_hum_2],
                  fmt='o', ms=err_bar_dot_size, color='darkslategrey')
ax[3, 0].scatter(dist_list, min_mstar_hum_2, s=scatter_dot_size, color='tab:green')

ax[2, 0].scatter([], [], s=scatter_dot_size, color='tab:purple', label='Most massive')
ax[2, 0].scatter([], [], s=scatter_dot_size, color='darkslategrey', label='Median value')
ax[2, 0].scatter([], [], s=scatter_dot_size, color='tab:green', label='Least massive')


ax[2, 1].scatter(dist_list, max_mstar_ml_1, s=scatter_dot_size, color='tab:purple')
ax[2, 1].errorbar(dist_list, median_mstar_ml_1,
                  yerr=[median_mstar_ml_1 - p16_mstar_ml_1,
                        p84_mstar_ml_1 - median_mstar_ml_1],
                  fmt='o', ms=err_bar_dot_size, color='darkslategrey')
ax[2, 1].scatter(dist_list, min_mstar_ml_1, s=scatter_dot_size, color='tab:green')

ax[3, 1].scatter(dist_list, max_mstar_ml_2, s=scatter_dot_size, color='tab:purple')
ax[3, 1].errorbar(dist_list, median_mstar_ml_2,
                  yerr=[median_mstar_ml_2 - p16_mstar_ml_2,
                        p84_mstar_ml_2 - median_mstar_ml_2],
                  fmt='o', ms=err_bar_dot_size, color='darkslategrey')
ax[3, 1].scatter(dist_list, min_mstar_ml_2, s=scatter_dot_size, color='tab:green')


ax[0, 0].set_title('Human Class 1', fontsize=fontsize+4)
ax[1, 0].set_title('Human Class 2', fontsize=fontsize+4)

ax[0, 1].set_title('ML Class 1', fontsize=fontsize+4)
ax[1, 1].set_title('ML Class 2', fontsize=fontsize+4)

ax[2, 0].set_title('Human Class 1', fontsize=fontsize+4)
ax[3, 0].set_title('Human Class 2', fontsize=fontsize+4)

ax[2, 1].set_title('ML Class 1', fontsize=fontsize+4)
ax[3, 1].set_title('ML Class 2', fontsize=fontsize+4)

ax[0, 0].legend(frameon=False, fontsize=fontsize - 6)
ax[2, 0].legend(frameon=False, fontsize=fontsize - 6)

min_y_val_abs = np.min([*ax[0, 0].get_ylim(), *ax[1, 0].get_ylim()])
max_y_val_abs = np.max([*ax[0, 0].get_ylim(), *ax[1, 0].get_ylim()])

ax[0, 0].set_ylim(min_y_val_abs, max_y_val_abs)
ax[1, 0].set_ylim(min_y_val_abs, max_y_val_abs)

ax[2, 0].set_yscale('log')
ax[3, 0].set_yscale('log')

min_y_val_m_star = np.min([*ax[2, 0].get_ylim(), *ax[3, 0].get_ylim(), *ax[2, 1].get_ylim(), *ax[3, 1].get_ylim()])
max_y_val_m_star = np.max([*ax[2, 0].get_ylim(), *ax[3, 0].get_ylim()])

print('min_y_val_m_star ', min_y_val_m_star)
print('max_y_val_m_star ', max_y_val_m_star)

print(np.min(min_mstar_hum_1))
print(np.min(min_mstar_hum_2))
print(np.min(min_mstar_ml_1))
print(np.min(min_mstar_ml_2))

ax[2, 0].set_ylim(min_y_val_m_star, max_y_val_m_star)
ax[3, 0].set_ylim(min_y_val_m_star, max_y_val_m_star)

ax[0, 0].invert_yaxis()
ax[1, 0].invert_yaxis()

ax[0, 0].set_ylabel('Abs. V-mag', fontsize=fontsize)
ax[1, 0].set_ylabel('Abs. V-mag', fontsize=fontsize)
ax[2, 0].set_ylabel(r'log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
ax[3, 0].set_ylabel(r'log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
ax[3, 0].set_xlabel('Distance [Mpc]', fontsize=fontsize)
ax[3, 1].set_xlabel('Distance [Mpc]', fontsize=fontsize)

ax[0, 0].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[3, 0].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)
ax[3, 1].tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in', labelsize=fontsize)

# plt.tight_layout()
# plt.subplots_adjust(hspace=0.01, wspace=0.01)
fig.subplots_adjust(left=0.055, bottom=0.05, right=0.995, top=0.975, wspace=0.005, hspace=0.085)
plt.savefig('plot_output/mag_mstar_c1_c2.png')
plt.savefig('plot_output/mag_mstar_c1_c2.pdf')

