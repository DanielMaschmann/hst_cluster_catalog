import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
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

target_list = ['ngc3621']

catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


target = 'ngc3621'
dist = catalog_access.dist_dict[target]['dist']

index_12_ml = catalog_access.get_hst_cc_index(target=target, classify='ml')
candidate_index_12_ml = catalog_access.get_hst_cc_phangs_candidate_id(target=target, classify='ml')
cluster_index_12_ml = catalog_access.get_hst_cc_phangs_cluster_id(target=target, classify='ml')
ci_12_ml = catalog_access.get_hst_cc_ci(target=target, classify='ml')
hum_class_12_ml = catalog_access.get_hst_cc_class_human(target=target, classify='ml')
ml_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
mass_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
ebv_12_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
color_vi_12_ml = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
color_ub_12_ml = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
v_mag_12_ml = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band='F555W')
abs_v_mag_12_ml = hf.conv_mag2abs_mag(mag=v_mag_12_ml, dist=float(dist))
ra_12_ml, dec_12_ml = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')

index_3_ml = catalog_access.get_hst_cc_index(target=target, cluster_class='class3', classify='ml')
candidate_index_3_ml = catalog_access.get_hst_cc_phangs_candidate_id(target=target, cluster_class='class3', classify='ml')
cluster_index_3_ml = catalog_access.get_hst_cc_phangs_cluster_id(target=target, cluster_class='class3', classify='ml')
ci_3_ml = catalog_access.get_hst_cc_ci(target=target, cluster_class='class3', classify='ml')
hum_class_3_ml = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3', classify='ml')
ml_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, cluster_class='class3', classify='ml')
mass_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3', classify='ml')
age_3_ml = catalog_access.get_hst_cc_age(target=target, cluster_class='class3', classify='ml')
ebv_3_ml = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3', classify='ml')
color_vi_3_ml = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3', classify='ml')
color_ub_3_ml = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3', classify='ml')
v_mag_3_ml = catalog_access.get_hst_cc_band_vega_mag(target=target, cluster_class='class3', classify='ml', band='F555W')
abs_v_mag_3_ml = hf.conv_mag2abs_mag(mag=v_mag_3_ml, dist=float(dist))
ra_3_ml, dec_3_ml = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3', classify='ml')



index_ml = np.concatenate([index_12_ml, index_3_ml])
candidate_index_ml = np.concatenate([candidate_index_12_ml, candidate_index_3_ml])
cluster_index_ml = np.concatenate([cluster_index_12_ml, cluster_index_3_ml])
ci_ml = np.concatenate([ci_12_ml, ci_3_ml])
ml_class_ml = np.concatenate([ml_class_12_ml, ml_class_3_ml])
hum_class_ml = np.concatenate([hum_class_12_ml, hum_class_3_ml])
color_vi_ml = np.concatenate([color_vi_12_ml, color_vi_3_ml])
age_ml = np.concatenate([age_12_ml, age_3_ml])
abs_v_mag_ml = np.concatenate([abs_v_mag_12_ml, abs_v_mag_3_ml])
ebv_ml = np.concatenate([ebv_12_ml, ebv_3_ml])
mass_ml = np.concatenate([mass_12_ml, mass_3_ml])
ra_ml = np.concatenate([ra_12_ml, ra_3_ml])
dec_ml = np.concatenate([dec_12_ml, dec_3_ml])

mask_problematic_values = (age_ml > 10) & (ebv_ml > 1.4)
print(sum(mask_problematic_values))

visualization_access = PhotVisualize(
                            target_name=target,
                            hst_data_path=hst_data_path,
                            nircam_data_path=nircam_data_path,
                            miri_data_path=miri_data_path,
                            hst_data_ver=hst_data_ver,
                            nircam_data_ver=nircam_data_ver,
                            miri_data_ver=miri_data_ver
                        )
visualization_access.load_hst_nircam_miri_bands(flux_unit='MJy/sr', load_err=False)

for ra, dec, index, candidate_id, cluster_id, ml_class, hum_class, age, ebv, mass, abs_v_mag, color_vi, ci in zip(
        ra_ml[mask_problematic_values],
        dec_ml[mask_problematic_values],
        index_ml[mask_problematic_values],
        candidate_index_ml[mask_problematic_values],
        cluster_index_ml[mask_problematic_values],
        ml_class_ml[mask_problematic_values],
        hum_class_ml[mask_problematic_values],
        age_ml[mask_problematic_values],
        ebv_ml[mask_problematic_values],
        mass_ml[mask_problematic_values],
        abs_v_mag_ml[mask_problematic_values],
        color_vi_ml[mask_problematic_values],
        ci_ml[mask_problematic_values]):

    str_line_1 = ('%s, dist=%.1f Mpc,         '
                  'INDEX = %i, '
                  'ID_PHANGS_CANDIDATE = %i, '
                  'ID_PHANGS_CLUSTERS = %i, '
                  'HUM_CLASS = %s, '
                  'ML_CLASS = %s, '
                  % (target, float(dist), index, candidate_id, cluster_id, hum_class, ml_class))
                  # add V-I color to the

    str_line_2 = (r'age= %i,    '
                  r'E(B-V)=%.2f,    '
                  r'log(M$_{*}$/M$_{\odot}$)=%.2f  '
                  r'M$_{\rm V}$ = %.1f   '
                  r'V-I = %.1f    '
                  r'CI = %.1f' %
                  (age, ebv, np.log10(mass), abs_v_mag, color_vi, ci))

    fig = visualization_access.plot_multi_band_artifacts(ra=ra, dec=dec,
                                                         str_line_1=str_line_1,
                                                         str_line_2=str_line_2,
                                                         cutout_size=5.0, circle_rad=0.2)

    fig.savefig('plot_output/ngc_3621_problematic_sed_fit/obj_in_%s_%i.png' % (target, index))
    plt.clf()
    plt.close("all")