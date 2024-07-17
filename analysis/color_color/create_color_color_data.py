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
                                                            sample_table_path=sample_table_path,
                                                            hst_cc_ver='phangs_hst_cc_dr4_cr3_ground_based_ha')

# get model
hdu_a_sol = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod_sol = hdu_a_sol[1].data
age_mod_sol = data_mod_sol['sfh.age']
flux_f275w_sol = data_mod_sol['F275W_UVIS_CHIP2']
flux_f336w_sol = data_mod_sol['F336W_UVIS_CHIP2']
flux_f438w_sol = data_mod_sol['F438W_UVIS_CHIP2']
flux_f555w_sol = data_mod_sol['F555W_UVIS_CHIP2']
flux_f814w_sol = data_mod_sol['F814W_UVIS_CHIP2']
mag_nuv_sol = hf.conv_mjy2vega(flux=flux_f275w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
mag_u_sol = hf.conv_mjy2vega(flux=flux_f336w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol = hf.conv_mjy2vega(flux=flux_f438w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_v_sol = hf.conv_mjy2vega(flux=flux_f555w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol = hf.conv_mjy2vega(flux=flux_f814w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
model_nuvu_sol = mag_nuv_sol - mag_u_sol
model_nuvb_sol = mag_nuv_sol - mag_b_sol
model_ub_sol = mag_u_sol - mag_b_sol
model_bv_sol = mag_b_sol - mag_v_sol
model_bi_sol = mag_b_sol - mag_i_sol
model_vi_sol = mag_v_sol - mag_i_sol


np.save('data_output/age_mod_sol.npy', age_mod_sol)
np.save('data_output/model_nuvu_sol.npy', model_nuvu_sol)
np.save('data_output/model_nuvb_sol.npy', model_nuvb_sol)
np.save('data_output/model_ub_sol.npy', model_ub_sol)
np.save('data_output/model_bv_sol.npy', model_bv_sol)
np.save('data_output/model_bi_sol.npy', model_bv_sol)
np.save('data_output/model_vi_sol.npy', model_vi_sol)


# get model
hdu_a_sol50 = fits.open('../cigale_model/sfh2exp/no_dust/sol_met_50/out/models-block-0.fits')
data_mod_sol50 = hdu_a_sol50[1].data
age_mod_sol50 = data_mod_sol50['sfh.age']
flux_f275w_sol50 = data_mod_sol50['F275W_UVIS_CHIP2']
flux_f336w_sol50 = data_mod_sol50['F336W_UVIS_CHIP2']
flux_f555w_sol50 = data_mod_sol50['F555W_UVIS_CHIP2']
flux_f814w_sol50 = data_mod_sol50['F814W_UVIS_CHIP2']
flux_f438w_sol50 = data_mod_sol50['F438W_UVIS_CHIP2']
mag_nuv_sol50 = hf.conv_mjy2vega(flux=flux_f275w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
mag_u_sol50 = hf.conv_mjy2vega(flux=flux_f336w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol50 = hf.conv_mjy2vega(flux=flux_f438w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_v_sol50 = hf.conv_mjy2vega(flux=flux_f555w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol50 = hf.conv_mjy2vega(flux=flux_f814w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
model_nuvu_sol50 = mag_nuv_sol50 - mag_u_sol50
model_nuvb_sol50 = mag_nuv_sol50 - mag_b_sol50
model_ub_sol50 = mag_u_sol50 - mag_b_sol50
model_bv_sol50 = mag_b_sol50 - mag_v_sol50
model_bi_sol50 = mag_b_sol50 - mag_i_sol50
model_vi_sol50 = mag_v_sol50 - mag_i_sol50

np.save('data_output/age_mod_sol50.npy', age_mod_sol50)
np.save('data_output/model_nuvu_sol50.npy', model_nuvu_sol50)
np.save('data_output/model_nuvb_sol50.npy', model_nuvb_sol50)
np.save('data_output/model_ub_sol50.npy', model_ub_sol50)
np.save('data_output/model_bv_sol50.npy', model_bv_sol50)
np.save('data_output/model_bi_sol50.npy', model_bi_sol50)
np.save('data_output/model_vi_sol50.npy', model_vi_sol50)


# get model
hdu_a_sol5 = fits.open('../cigale_model/sfh2exp/no_dust/sol_met_5/out/models-block-0.fits')
data_mod_sol5 = hdu_a_sol5[1].data
age_mod_sol5 = data_mod_sol5['sfh.age']
flux_f275w_sol5 = data_mod_sol5['F275W_UVIS_CHIP2']
flux_f336w_sol5 = data_mod_sol5['F336W_UVIS_CHIP2']
flux_f555w_sol5 = data_mod_sol5['F555W_UVIS_CHIP2']
flux_f814w_sol5 = data_mod_sol5['F814W_UVIS_CHIP2']
flux_f438w_sol5 = data_mod_sol5['F438W_UVIS_CHIP2']
mag_nuv_sol5 = hf.conv_mjy2vega(flux=flux_f275w_sol5, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
mag_u_sol5 = hf.conv_mjy2vega(flux=flux_f336w_sol5, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol5 = hf.conv_mjy2vega(flux=flux_f438w_sol5, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_v_sol5 = hf.conv_mjy2vega(flux=flux_f555w_sol5, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol5 = hf.conv_mjy2vega(flux=flux_f814w_sol5, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
model_nuvu_sol5 = mag_nuv_sol5 - mag_u_sol5
model_nuvb_sol5 = mag_nuv_sol5 - mag_b_sol5
model_ub_sol5 = mag_u_sol5 - mag_b_sol5
model_bv_sol5 = mag_b_sol5 - mag_v_sol5
model_bi_sol5 = mag_b_sol5 - mag_i_sol5
model_vi_sol5 = mag_v_sol5 - mag_i_sol5

np.save('data_output/age_mod_sol5.npy', age_mod_sol5)
np.save('data_output/model_nuvu_sol5.npy', model_nuvu_sol5)
np.save('data_output/model_nuvb_sol5.npy', model_nuvb_sol5)
np.save('data_output/model_ub_sol5.npy', model_ub_sol5)
np.save('data_output/model_bv_sol5.npy', model_bv_sol5)
np.save('data_output/model_bi_sol5.npy', model_bi_sol5)
np.save('data_output/model_vi_sol5.npy', model_vi_sol5)


# get model
hdu_a_sol_neb = fits.open('/home/benutzer/Documents/projects/hst_cluster_catalog/analysis/cigale_model/sfh2exp/no_dust/sol_met_gas/out/models-block-0.fits')
data_mod_sol_neb = hdu_a_sol_neb[1].data
# print(data_mod_sol_neb.names)

ew_h_alpha = data_mod_sol_neb['param.EW(656.3/1.0)']
age_mod_sol_neb = data_mod_sol_neb['sfh.age']
logu_mod_sol_neb = data_mod_sol_neb['nebular.logU']
ne_mod_sol_neb = data_mod_sol_neb['nebular.ne']

flux_f275w_sol_neb = data_mod_sol_neb['F275W_UVIS_CHIP2']
flux_f336w_sol_neb = data_mod_sol_neb['F336W_UVIS_CHIP2']
flux_f438w_sol_neb = data_mod_sol_neb['F438W_UVIS_CHIP2']
flux_f555w_sol_neb = data_mod_sol_neb['F555W_UVIS_CHIP2']
flux_f814w_sol_neb = data_mod_sol_neb['F814W_UVIS_CHIP2']


mag_nuv_sol_neb = hf.conv_mjy2vega(flux=flux_f275w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
mag_u_sol_neb = hf.conv_mjy2vega(flux=flux_f336w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol_neb = hf.conv_mjy2vega(flux=flux_f438w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_v_sol_neb = hf.conv_mjy2vega(flux=flux_f555w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol_neb = hf.conv_mjy2vega(flux=flux_f814w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
model_nuvu_sol_neb = mag_nuv_sol_neb - mag_u_sol_neb
model_nuvb_sol_neb = mag_nuv_sol_neb - mag_b_sol_neb
model_ub_sol_neb = mag_u_sol_neb - mag_b_sol_neb
model_bv_sol_neb = mag_b_sol_neb - mag_v_sol_neb
model_bi_sol_neb = mag_b_sol_neb - mag_i_sol_neb
model_vi_sol_neb = mag_v_sol_neb - mag_i_sol_neb


np.save('data_output/age_mod_sol_neb.npy', age_mod_sol_neb)
np.save('data_output/model_nuvu_sol_neb.npy', model_nuvu_sol_neb)
np.save('data_output/model_nuvb_sol_neb.npy', model_nuvb_sol_neb)
np.save('data_output/model_ub_sol_neb.npy', model_ub_sol_neb)
np.save('data_output/model_bv_sol_neb.npy', model_bv_sol_neb)
np.save('data_output/model_bi_sol_neb.npy', model_bi_sol_neb)
np.save('data_output/model_vi_sol_neb.npy', model_vi_sol_neb)



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
incl_hum = np.array([])
index_hum = np.array([])
phangs_id_hum = np.array([])
color_vi_hum = np.array([])
color_ub_hum = np.array([])
color_bv_hum = np.array([])
color_bi_hum = np.array([])
color_nuvu_hum = np.array([])
color_nuvb_hum = np.array([])
color_vi_err_hum = np.array([])
color_ub_err_hum = np.array([])
color_bv_err_hum = np.array([])
color_bi_err_hum = np.array([])
color_nuvu_err_hum = np.array([])
color_nuvb_err_hum = np.array([])

detect_nuv_hum = np.array([], dtype=bool)
detect_u_hum = np.array([], dtype=bool)
detect_b_hum = np.array([], dtype=bool)
detect_v_hum = np.array([], dtype=bool)
detect_i_hum = np.array([], dtype=bool)

abs_v_mag_hum = np.array([])


clcl_color_hum = np.array([])
age_hum = np.array([])
ebv_hum = np.array([])
mass_hum = np.array([])
ra_hum = np.array([])
dec_hum = np.array([])
x_hum = np.array([])
y_hum = np.array([])


target_name_ml = np.array([], dtype=str)
incl_ml = np.array([])
index_ml = np.array([])
phangs_id_ml = np.array([])
color_vi_ml = np.array([])
color_ub_ml = np.array([])
color_bv_ml = np.array([])
color_bi_ml = np.array([])
color_nuvu_ml = np.array([])
color_nuvb_ml = np.array([])
color_vi_err_ml = np.array([])
color_ub_err_ml = np.array([])
color_bv_err_ml = np.array([])
color_bi_err_ml = np.array([])
color_nuvu_err_ml = np.array([])
color_nuvb_err_ml = np.array([])

detect_nuv_ml = np.array([], dtype=bool)
detect_u_ml = np.array([], dtype=bool)
detect_b_ml = np.array([], dtype=bool)
detect_v_ml = np.array([], dtype=bool)
detect_i_ml = np.array([], dtype=bool)

abs_v_mag_ml = np.array([])

clcl_color_ml = np.array([])
clcl_qual_color_ml = np.array([])
age_ml = np.array([])
ebv_ml = np.array([])
mass_ml = np.array([])
ra_ml = np.array([])
dec_ml = np.array([])
x_ml = np.array([])
y_ml = np.array([])
mag_mask_ml = np.array([], dtype=bool)

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
    phangs_id_hum_12 = catalog_access.get_hst_cc_phangs_cluster_id(target=target)
    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_bv_hum_12 = catalog_access.get_hst_color_bv_vega(target=target)
    color_bi_hum_12 = catalog_access.get_hst_color_bi_vega(target=target)
    color_nuvu_hum_12 = catalog_access.get_hst_color_nuvu_vega(target=target)
    color_nuvb_hum_12 = catalog_access.get_hst_color_nuvb_vega(target=target)

    color_vi_err_hum_12 = catalog_access.get_hst_color_vi_err(target=target)
    color_ub_err_hum_12 = catalog_access.get_hst_color_ub_err(target=target)
    color_bv_err_hum_12 = catalog_access.get_hst_color_bv_err(target=target)
    color_bi_err_hum_12 = catalog_access.get_hst_color_bi_err(target=target)
    color_nuvu_err_hum_12 = catalog_access.get_hst_color_nuvu_err(target=target)
    color_nuvb_err_hum_12 = catalog_access.get_hst_color_nuvb_err(target=target)

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

    v_mag_hum_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    abs_v_mag_hum_12 = hf.conv_mag2abs_mag(mag=v_mag_hum_12, dist=dist)

    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    ebv_hum_12 = catalog_access.get_hst_cc_ebv(target=target)
    mass_hum_12 = catalog_access.get_hst_cc_stellar_m(target=target)
    ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
    x_hum_12, y_hum_12 = catalog_access.get_hst_cc_coords_pix(target=target)


    index_hum_3 = catalog_access.get_hst_cc_index(target=target, cluster_class='class3')
    phangs_id_hum_3 = catalog_access.get_hst_cc_phangs_cluster_id(target=target, cluster_class='class3')
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_bv_hum_3 = catalog_access.get_hst_color_bv_vega(target=target, cluster_class='class3')
    color_bi_hum_3 = catalog_access.get_hst_color_bi_vega(target=target, cluster_class='class3')
    color_nuvu_hum_3 = catalog_access.get_hst_color_nuvu_vega(target=target, cluster_class='class3')
    color_nuvb_hum_3 = catalog_access.get_hst_color_nuvb_vega(target=target, cluster_class='class3')

    color_vi_err_hum_3 = catalog_access.get_hst_color_vi_err(target=target, cluster_class='class3')
    color_ub_err_hum_3 = catalog_access.get_hst_color_ub_err(target=target, cluster_class='class3')
    color_bv_err_hum_3 = catalog_access.get_hst_color_bv_err(target=target, cluster_class='class3')
    color_bi_err_hum_3 = catalog_access.get_hst_color_bi_err(target=target, cluster_class='class3')
    color_nuvu_err_hum_3 = catalog_access.get_hst_color_nuvu_err(target=target, cluster_class='class3')
    color_nuvb_err_hum_3 = catalog_access.get_hst_color_nuvb_err(target=target, cluster_class='class3')

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

    v_mag_hum_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', cluster_class='class3')
    abs_v_mag_hum_3 = hf.conv_mag2abs_mag(mag=v_mag_hum_3, dist=dist)

    age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_hum_3 = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    mass_hum_3 = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
    x_hum_3, y_hum_3 = catalog_access.get_hst_cc_coords_pix(target=target, cluster_class='class3')

    color_vi_hum = np.concatenate([color_vi_hum, color_vi_hum_12, color_vi_hum_3])
    color_ub_hum = np.concatenate([color_ub_hum, color_ub_hum_12, color_ub_hum_3])
    color_bv_hum = np.concatenate([color_bv_hum, color_bv_hum_12, color_bv_hum_3])
    color_bi_hum = np.concatenate([color_bi_hum, color_bi_hum_12, color_bi_hum_3])
    color_nuvu_hum = np.concatenate([color_nuvu_hum, color_nuvu_hum_12, color_nuvu_hum_3])
    color_nuvb_hum = np.concatenate([color_nuvb_hum, color_nuvb_hum_12, color_nuvb_hum_3])

    color_vi_err_hum = np.concatenate([color_vi_err_hum, color_vi_err_hum_12, color_vi_err_hum_3])
    color_ub_err_hum = np.concatenate([color_ub_err_hum, color_ub_err_hum_12, color_ub_err_hum_3])
    color_bv_err_hum = np.concatenate([color_bv_err_hum, color_bv_err_hum_12, color_bv_err_hum_3])
    color_bi_err_hum = np.concatenate([color_bi_err_hum, color_bi_err_hum_12, color_bi_err_hum_3])
    color_nuvu_err_hum = np.concatenate([color_nuvu_err_hum, color_nuvu_err_hum_12, color_nuvu_err_hum_3])
    color_nuvb_err_hum = np.concatenate([color_nuvb_err_hum, color_nuvb_err_hum_12, color_nuvb_err_hum_3])

    detect_nuv_hum = np.concatenate([detect_nuv_hum, detect_nuv_hum_12, detect_nuv_hum_3])
    detect_u_hum = np.concatenate([detect_u_hum, detect_u_hum_12, detect_u_hum_3])
    detect_b_hum = np.concatenate([detect_b_hum, detect_b_hum_12, detect_b_hum_3])
    detect_v_hum = np.concatenate([detect_v_hum, detect_v_hum_12, detect_v_hum_3])
    detect_i_hum = np.concatenate([detect_i_hum, detect_i_hum_12, detect_i_hum_3])

    abs_v_mag_hum = np.concatenate([abs_v_mag_hum, abs_v_mag_hum_12, abs_v_mag_hum_3])

    index_hum = np.concatenate([index_hum, index_hum_12, index_hum_3])
    phangs_id_hum = np.concatenate([phangs_id_hum, phangs_id_hum_12, phangs_id_hum_3])
    clcl_color_hum = np.concatenate([clcl_color_hum, cluster_class_hum_12, cluster_class_hum_3])
    target_name_hum_12 = np.array([target]*len(cluster_class_hum_12))
    target_name_hum_3 = np.array([target]*len(cluster_class_hum_3))
    target_name_hum = np.concatenate([target_name_hum, target_name_hum_12, target_name_hum_3])
    incl_hum_12 = np.array([inclination]*len(cluster_class_hum_12))
    incl_hum_3 = np.array([inclination]*len(cluster_class_hum_3))
    incl_hum = np.concatenate([incl_hum, incl_hum_12, incl_hum_3])


    age_hum = np.concatenate([age_hum, age_hum_12, age_hum_3])
    ebv_hum = np.concatenate([ebv_hum, ebv_hum_12, ebv_hum_3])
    mass_hum = np.concatenate([mass_hum, mass_hum_12, mass_hum_3])
    ra_hum = np.concatenate([ra_hum, ra_hum_12, ra_hum_3])
    dec_hum = np.concatenate([dec_hum, dec_hum_12, dec_hum_3])
    x_hum = np.concatenate([x_hum, x_hum_12, x_hum_3])
    y_hum = np.concatenate([y_hum, y_hum_12, y_hum_3])




    index_ml_12= catalog_access.get_hst_cc_index(target=target, classify='ml')
    phangs_id_ml_12= catalog_access.get_hst_cc_phangs_cluster_id(target=target, classify='ml')
    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_bv_ml_12 = catalog_access.get_hst_color_bv_vega(target=target, classify='ml')
    color_bi_ml_12 = catalog_access.get_hst_color_bi_vega(target=target, classify='ml')
    color_nuvu_ml_12 = catalog_access.get_hst_color_nuvu_vega(target=target, classify='ml')
    color_nuvb_ml_12 = catalog_access.get_hst_color_nuvb_vega(target=target, classify='ml')

    color_vi_err_ml_12 = catalog_access.get_hst_color_vi_err(target=target, classify='ml')
    color_ub_err_ml_12 = catalog_access.get_hst_color_ub_err(target=target, classify='ml')
    color_bv_err_ml_12 = catalog_access.get_hst_color_bv_err(target=target, classify='ml')
    color_bi_err_ml_12 = catalog_access.get_hst_color_bi_err(target=target, classify='ml')
    color_nuvu_err_ml_12 = catalog_access.get_hst_color_nuvu_err(target=target, classify='ml')
    color_nuvb_err_ml_12 = catalog_access.get_hst_color_nuvb_err(target=target, classify='ml')

    detect_nuv_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F275W') > 0) &
                         (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F275W') >
                          catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F275W')))
    detect_u_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F336W')))
    detect_b_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band=b_band)))
    detect_v_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F555W')))
    detect_i_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F814W')))

    v_mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band='F555W')
    abs_v_mag_ml_12 = hf.conv_mag2abs_mag(mag=v_mag_ml_12, dist=dist)

    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_ml_12 = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    mass_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    x_ml_12, y_ml_12 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml')

    index_ml_3 = catalog_access.get_hst_cc_index(target=target, classify='ml', cluster_class='class3')
    phangs_id_ml_3 = catalog_access.get_hst_cc_phangs_cluster_id(target=target, classify='ml', cluster_class='class3')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_bv_ml_3 = catalog_access.get_hst_color_bv_vega(target=target, classify='ml', cluster_class='class3')
    color_bi_ml_3 = catalog_access.get_hst_color_bi_vega(target=target, classify='ml', cluster_class='class3')
    color_nuvu_ml_3 = catalog_access.get_hst_color_nuvu_vega(target=target, classify='ml', cluster_class='class3')
    color_nuvb_ml_3 = catalog_access.get_hst_color_nuvb_vega(target=target, classify='ml', cluster_class='class3')

    color_vi_err_ml_3 = catalog_access.get_hst_color_vi_err(target=target, classify='ml', cluster_class='class3')
    color_ub_err_ml_3 = catalog_access.get_hst_color_ub_err(target=target, classify='ml', cluster_class='class3')
    color_bv_err_ml_3 = catalog_access.get_hst_color_bv_err(target=target, classify='ml', cluster_class='class3')
    color_bi_err_ml_3 = catalog_access.get_hst_color_bi_err(target=target, classify='ml', cluster_class='class3')
    color_nuvu_err_ml_3 = catalog_access.get_hst_color_nuvu_err(target=target, classify='ml', cluster_class='class3')
    color_nuvb_err_ml_3 = catalog_access.get_hst_color_nuvb_err(target=target, classify='ml', cluster_class='class3')

    detect_nuv_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F275W') > 0) &
                         (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F275W') >
                          catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band='F275W')))
    detect_u_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F336W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F336W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band='F336W')))
    detect_b_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band=b_band)))
    detect_v_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band='F555W')))
    detect_i_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F814W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F814W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band='F814W')))

    v_mag_ml_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band='F555W', cluster_class='class3')
    abs_v_mag_ml_3 = hf.conv_mag2abs_mag(mag=v_mag_ml_3, dist=dist)

    age_ml_3 = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_ml_3 = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    mass_ml_3 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
    ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    x_ml_3, y_ml_3 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml', cluster_class='class3')

    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    color_bv_ml = np.concatenate([color_bv_ml, color_bv_ml_12, color_bv_ml_3])
    color_bi_ml = np.concatenate([color_bi_ml, color_bi_ml_12, color_bi_ml_3])
    color_nuvu_ml = np.concatenate([color_nuvu_ml, color_nuvu_ml_12, color_nuvu_ml_3])
    color_nuvb_ml = np.concatenate([color_nuvb_ml, color_nuvb_ml_12, color_nuvb_ml_3])

    color_vi_err_ml = np.concatenate([color_vi_err_ml, color_vi_err_ml_12, color_vi_err_ml_3])
    color_ub_err_ml = np.concatenate([color_ub_err_ml, color_ub_err_ml_12, color_ub_err_ml_3])
    color_bv_err_ml = np.concatenate([color_bv_err_ml, color_bv_err_ml_12, color_bv_err_ml_3])
    color_bi_err_ml = np.concatenate([color_bi_err_ml, color_bi_err_ml_12, color_bi_err_ml_3])
    color_nuvu_err_ml = np.concatenate([color_nuvu_err_ml, color_nuvu_err_ml_12, color_nuvu_err_ml_3])
    color_nuvb_err_ml = np.concatenate([color_nuvb_err_ml, color_nuvb_err_ml_12, color_nuvb_err_ml_3])

    detect_nuv_ml = np.concatenate([detect_nuv_ml, detect_nuv_ml_12, detect_nuv_ml_3])
    detect_u_ml = np.concatenate([detect_u_ml, detect_u_ml_12, detect_u_ml_3])
    detect_b_ml = np.concatenate([detect_b_ml, detect_b_ml_12, detect_b_ml_3])
    detect_v_ml = np.concatenate([detect_v_ml, detect_v_ml_12, detect_v_ml_3])
    detect_i_ml = np.concatenate([detect_i_ml, detect_i_ml_12, detect_i_ml_3])

    abs_v_mag_ml = np.concatenate([abs_v_mag_ml, abs_v_mag_ml_12, abs_v_mag_ml_3])

    index_ml = np.concatenate([index_ml, index_ml_12, index_ml_3])
    phangs_id_ml = np.concatenate([phangs_id_ml, phangs_id_ml_12, phangs_id_ml_3])
    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    clcl_qual_color_ml = np.concatenate([clcl_qual_color_ml, cluster_class_qual_ml_12, cluster_class_qual_ml_3])
    target_name_ml_12 = np.array([target]*len(cluster_class_ml_12))
    target_name_ml_3 = np.array([target]*len(cluster_class_ml_3))
    target_name_ml = np.concatenate([target_name_ml, target_name_ml_12, target_name_ml_3])
    incl_ml_12 = np.array([inclination]*len(cluster_class_ml_12))
    incl_ml_3 = np.array([inclination]*len(cluster_class_ml_3))
    incl_ml = np.concatenate([incl_ml, incl_ml_12, incl_ml_3])

    age_ml = np.concatenate([age_ml, age_ml_12, age_ml_3])
    ebv_ml = np.concatenate([ebv_ml, ebv_ml_12, ebv_ml_3])
    mass_ml = np.concatenate([mass_ml, mass_ml_12, mass_ml_3])
    ra_ml = np.concatenate([ra_ml, ra_ml_12, ra_ml_3])
    dec_ml = np.concatenate([dec_ml, dec_ml_12, dec_ml_3])
    x_ml = np.concatenate([x_ml, x_ml_12, x_ml_3])
    y_ml = np.concatenate([y_ml, y_ml_12, y_ml_3])


    print('max ebv', np.max(np.concatenate([ebv_ml_12, ebv_ml_3])))

    # now get magnitude cuts
    v_mag_hum_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    v_mag_hum_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', cluster_class='class3')
    abs_v_mag_hum_12 = hf.conv_mag2abs_mag(mag=v_mag_hum_12, dist=dist)
    abs_v_mag_hum_3 = hf.conv_mag2abs_mag(mag=v_mag_hum_3, dist=dist)
    v_mag_hum = np.concatenate([v_mag_hum_12, v_mag_hum_3])

    v_mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', classify='ml')
    v_mag_ml_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', classify='ml', cluster_class='class3')
    abs_v_mag_ml_12 = hf.conv_mag2abs_mag(mag=v_mag_ml_12, dist=dist)
    abs_v_mag_ml_3 = hf.conv_mag2abs_mag(mag=v_mag_ml_3, dist=dist)
    v_mag_ml = np.concatenate([v_mag_ml_12, v_mag_ml_3])

    max_hum_mag = np.nanmax(v_mag_hum)
    selected_mag_ml = v_mag_ml < max_hum_mag

    mag_mask_ml = np.concatenate([mag_mask_ml, selected_mag_ml])

np.save('data_output/target_name_hum.npy', target_name_hum)
np.save('data_output/incl_hum.npy', incl_hum)
np.save('data_output/color_vi_hum.npy', color_vi_hum)
np.save('data_output/color_ub_hum.npy', color_ub_hum)
np.save('data_output/color_bv_hum.npy', color_bv_hum)
np.save('data_output/color_bi_hum.npy', color_bi_hum)
np.save('data_output/color_nuvu_hum.npy', color_nuvu_hum)
np.save('data_output/color_nuvb_hum.npy', color_nuvb_hum)
np.save('data_output/color_vi_err_hum.npy', color_vi_err_hum)
np.save('data_output/color_ub_err_hum.npy', color_ub_err_hum)
np.save('data_output/color_bv_err_hum.npy', color_bv_err_hum)
np.save('data_output/color_bi_err_hum.npy', color_bi_err_hum)
np.save('data_output/color_nuvu_err_hum.npy', color_nuvu_err_hum)
np.save('data_output/color_nuvb_err_hum.npy', color_nuvb_err_hum)
np.save('data_output/detect_nuv_hum.npy', detect_nuv_hum)
np.save('data_output/detect_u_hum.npy', detect_u_hum)
np.save('data_output/detect_b_hum.npy', detect_b_hum)
np.save('data_output/detect_v_hum.npy', detect_v_hum)
np.save('data_output/detect_i_hum.npy', detect_i_hum)
np.save('data_output/abs_v_mag_hum.npy', abs_v_mag_hum)
np.save('data_output/index_hum.npy', index_hum)
np.save('data_output/phangs_id_hum.npy', phangs_id_hum)
np.save('data_output/clcl_color_hum.npy', clcl_color_hum)
np.save('data_output/age_hum.npy', age_hum)
np.save('data_output/ebv_hum.npy', ebv_hum)
np.save('data_output/mass_hum.npy', mass_hum)
np.save('data_output/ra_hum.npy', ra_hum)
np.save('data_output/dec_hum.npy', dec_hum)
np.save('data_output/x_hum.npy', x_hum)
np.save('data_output/y_hum.npy', y_hum)


np.save('data_output/target_name_ml.npy', target_name_ml)
np.save('data_output/incl_ml.npy', incl_ml)
np.save('data_output/color_vi_ml.npy', color_vi_ml)
np.save('data_output/color_ub_ml.npy', color_ub_ml)
np.save('data_output/color_bv_ml.npy', color_bv_ml)
np.save('data_output/color_bi_ml.npy', color_bi_ml)
np.save('data_output/color_nuvu_ml.npy', color_nuvu_ml)
np.save('data_output/color_nuvb_ml.npy', color_nuvb_ml)
np.save('data_output/color_vi_err_ml.npy', color_vi_err_ml)
np.save('data_output/color_ub_err_ml.npy', color_ub_err_ml)
np.save('data_output/color_bv_err_ml.npy', color_bv_err_ml)
np.save('data_output/color_bi_err_ml.npy', color_bi_err_ml)
np.save('data_output/color_nuvu_err_ml.npy', color_nuvu_err_ml)
np.save('data_output/color_nuvb_err_ml.npy', color_nuvb_err_ml)
np.save('data_output/detect_nuv_ml.npy', detect_nuv_ml)
np.save('data_output/detect_u_ml.npy', detect_u_ml)
np.save('data_output/detect_b_ml.npy', detect_b_ml)
np.save('data_output/detect_v_ml.npy', detect_v_ml)
np.save('data_output/detect_i_ml.npy', detect_i_ml)
np.save('data_output/abs_v_mag_ml.npy', abs_v_mag_ml)
np.save('data_output/index_ml.npy', index_ml)
np.save('data_output/phangs_id_ml.npy', phangs_id_ml)
np.save('data_output/clcl_color_ml.npy', clcl_color_ml)
np.save('data_output/clcl_qual_color_ml.npy', clcl_qual_color_ml)
np.save('data_output/age_ml.npy', age_ml)
np.save('data_output/ebv_ml.npy', ebv_ml)
np.save('data_output/mass_ml.npy', mass_ml)
np.save('data_output/ra_ml.npy', ra_ml)
np.save('data_output/dec_ml.npy', dec_ml)
np.save('data_output/x_ml.npy', x_ml)
np.save('data_output/y_ml.npy', y_ml)
np.save('data_output/mag_mask_ml.npy', mag_mask_ml)


print(np.unique(clcl_color_hum))
print(np.unique(clcl_color_ml))
print(np.unique(clcl_qual_color_ml))

