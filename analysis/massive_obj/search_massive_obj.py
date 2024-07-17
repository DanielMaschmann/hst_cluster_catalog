import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from cluster_cat_dr.visualization_tool import PhotVisualize
from scipy.spatial import ConvexHull
from photometry_tools import helper_func as hf


vi_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/vi_hull_young_ubvi_hum_3.npy')
ub_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/ub_hull_young_ubvi_hum_3.npy')

hull_young_hum = ConvexHull(np.array([vi_hull_young_ubvi_hum_3, ub_hull_young_ubvi_hum_3]).T)


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


catalog_access.load_hst_cc_list(target_list=target_list, classify='human')


for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    index_12_hum = catalog_access.get_hst_cc_index(target=target)
    hum_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    ml_class_12_hum = catalog_access.get_hst_cc_class_ml_vgg(target=target)
    mass_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    ebv_12_hum = catalog_access.get_hst_cc_ebv(target=target)
    color_vi_12_hum = catalog_access.get_hst_color_vi_vega(target=target)
    color_ub_12_hum = catalog_access.get_hst_color_ub_vega(target=target)
    v_mag_12_hum = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    abs_v_mag_12_hum = hf.conv_mag2abs_mag(mag=v_mag_12_hum, dist=float(dist))
    ra_12_hum, dec_12_hum = catalog_access.get_hst_cc_coords_world(target=target)

    in_hull_young_ml = hf.points_in_hull(np.array([color_vi_12_hum, color_ub_12_hum]).T, hull_young_hum)

    mask_young_massive_obj = (mass_12_hum > 1e5) & (age_12_hum < 2) & (hum_class_12_hum == 1) & (in_hull_young_ml)
    print(sum(mask_young_massive_obj))

    if sum(mask_young_massive_obj) > 0:
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

        for ra, dec, cluster_id, ml_class, hum_class, age, ebv, mass, abs_v_mag in zip(ra_12_hum[mask_young_massive_obj],
                                                                      dec_12_hum[mask_young_massive_obj],
                                                                      index_12_hum[mask_young_massive_obj],
                                                                      ml_class_12_hum[mask_young_massive_obj],
                                                                      hum_class_12_hum[mask_young_massive_obj],
                                                                      age_12_hum[mask_young_massive_obj],
                                                                      ebv_12_hum[mask_young_massive_obj],
                                                                            mass_12_hum[mask_young_massive_obj],
                                                                            abs_v_mag_12_hum[mask_young_massive_obj]):

            str_line_1 = ('%s, dist=%.1f Mpc,         '
                          'ID_PHANGS_CLUSTERS_v1p2 = %i, '
                          'HUM_CLASS = %s, '
                          'ML_CLASS = %s, '
                          % (target, float(dist), cluster_id, hum_class, ml_class))
                          # add V-I color to the
            str_line_2 = (r'age= %i,    E(B-V)=%.2f,    log(M$_{*}$/M$_{\odot}$)=%.2f  M$_{\rm V}$ = %.1f' %
                          (age, ebv, np.log10(mass), abs_v_mag))

            fig = visualization_access.plot_multi_band_artifacts(ra=ra, dec=dec,
                                                                 str_line_1=str_line_1,
                                                                 str_line_2=str_line_2,
                                                                 cutout_size=5.0, circle_rad=0.2)


            fig.savefig('plot_output/obj_in_%s_%i.png' % (target, cluster_id))
            plt.clf()
            plt.close("all")