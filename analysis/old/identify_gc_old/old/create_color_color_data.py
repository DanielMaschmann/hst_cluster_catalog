import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from matplotlib.patches import Ellipse
import matplotlib
from matplotlib.colorbar import ColorbarBase
from matplotlib.collections import LineCollection
import dust_tools.extinction_tools
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from lmfit import Model, Parameters


def gauss2d(x, y, x0, y0, sig_x, sig_y):
    expo = -(((x - x0)**2)/(2 * sig_x**2) + ((y - y0)**2)/(2 * sig_y**2))
    norm_amp = 1 / (2 * np.pi * sig_x * sig_y)
    return norm_amp * np.exp(expo)




cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

# get model
hdu_a_sol = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod_sol = hdu_a_sol[1].data
age_mod_sol = data_mod_sol['sfh.age']
flux_f555w_sol = data_mod_sol['F555W_UVIS_CHIP2']
flux_f814w_sol = data_mod_sol['F814W_UVIS_CHIP2']
flux_f336w_sol = data_mod_sol['F336W_UVIS_CHIP2']
flux_f438w_sol = data_mod_sol['F438W_UVIS_CHIP2']
mag_v_sol = hf.conv_mjy2vega(flux=flux_f555w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol = hf.conv_mjy2vega(flux=flux_f814w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_sol = hf.conv_mjy2vega(flux=flux_f336w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol = hf.conv_mjy2vega(flux=flux_f438w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_sol = mag_v_sol - mag_i_sol
model_ub_sol = mag_u_sol - mag_b_sol
model_bv_sol = mag_b_sol - mag_v_sol

np.save('data_output/model_vi_sol.npy', model_vi_sol)
np.save('data_output/model_ub_sol.npy', model_ub_sol)
np.save('data_output/model_bv_sol.npy', model_bv_sol)



# get model
hdu_a_sol50 = fits.open('../cigale_model/sfh2exp/no_dust/sol_met_50/out/models-block-0.fits')
data_mod_sol50 = hdu_a_sol50[1].data
age_mod_sol50 = data_mod_sol50['sfh.age']
flux_f555w_sol50 = data_mod_sol50['F555W_UVIS_CHIP2']
flux_f814w_sol50 = data_mod_sol50['F814W_UVIS_CHIP2']
flux_f336w_sol50 = data_mod_sol50['F336W_UVIS_CHIP2']
flux_f438w_sol50 = data_mod_sol50['F438W_UVIS_CHIP2']
mag_v_sol50 = hf.conv_mjy2vega(flux=flux_f555w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol50 = hf.conv_mjy2vega(flux=flux_f814w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_sol50 = hf.conv_mjy2vega(flux=flux_f336w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol50 = hf.conv_mjy2vega(flux=flux_f438w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_sol50 = mag_v_sol50 - mag_i_sol50
model_ub_sol50 = mag_u_sol50 - mag_b_sol50
model_bv_sol50 = mag_b_sol50 - mag_v_sol50

np.save('data_output/model_vi_sol50.npy', model_vi_sol50)
np.save('data_output/model_ub_sol50.npy', model_ub_sol50)
np.save('data_output/model_bv_sol50.npy', model_bv_sol50)


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


color_vi_hum = np.array([])
color_ub_hum = np.array([])
color_bv_hum = np.array([])
color_vi_err_hum = np.array([])
color_ub_err_hum = np.array([])
color_bv_err_hum = np.array([])
detect_vi_hum = np.array([], dtype=bool)
detect_ub_hum = np.array([], dtype=bool)
detect_bv_hum = np.array([], dtype=bool)
clcl_color_hum = np.array([])
age_hum = np.array([])
ebv_hum = np.array([])

color_vi_ml = np.array([])
color_ub_ml = np.array([])
color_bv_ml = np.array([])
color_vi_err_ml = np.array([])
color_ub_err_ml = np.array([])
color_bv_err_ml = np.array([])
detect_vi_ml = np.array([], dtype=bool)
detect_ub_ml = np.array([], dtype=bool)
detect_bv_ml = np.array([], dtype=bool)
clcl_color_ml = np.array([])
clcl_qual_color_ml = np.array([])
age_ml = np.array([])
ebv_ml = np.array([])


for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'


    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_bv_hum_12 = catalog_access.get_hst_color_bv_vega(target=target)
    color_vi_err_hum_12 = catalog_access.get_hst_color_vi_err(target=target)
    color_ub_err_hum_12 = catalog_access.get_hst_color_ub_err(target=target)
    color_bv_err_hum_12 = catalog_access.get_hst_color_bv_err(target=target)
    detect_vi_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0))
    detect_ub_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0))
    detect_bv_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0))
    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    ebv_hum_12 = catalog_access.get_hst_cc_ebv(target=target)

    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_bv_hum_3 = catalog_access.get_hst_color_bv_vega(target=target, cluster_class='class3')
    color_vi_err_hum_3 = catalog_access.get_hst_color_vi_err(target=target, cluster_class='class3')
    color_ub_err_hum_3 = catalog_access.get_hst_color_ub_err(target=target, cluster_class='class3')
    color_bv_err_hum_3 = catalog_access.get_hst_color_bv_err(target=target, cluster_class='class3')
    detect_vi_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F814W') > 0))
    detect_ub_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) > 0))
    detect_bv_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') > 0))
    age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_hum_3 = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')

    color_vi_hum = np.concatenate([color_vi_hum, color_vi_hum_12, color_vi_hum_3])
    color_ub_hum = np.concatenate([color_ub_hum, color_ub_hum_12, color_ub_hum_3])
    color_bv_hum = np.concatenate([color_bv_hum, color_bv_hum_12, color_bv_hum_3])

    color_vi_err_hum = np.concatenate([color_vi_err_hum, color_vi_err_hum_12, color_vi_err_hum_3])
    color_ub_err_hum = np.concatenate([color_ub_err_hum, color_ub_err_hum_12, color_ub_err_hum_3])
    color_bv_err_hum = np.concatenate([color_bv_err_hum, color_bv_err_hum_12, color_bv_err_hum_3])

    detect_vi_hum = np.concatenate([detect_vi_hum, detect_vi_hum_12, detect_vi_hum_3])
    detect_ub_hum = np.concatenate([detect_ub_hum, detect_ub_hum_12, detect_ub_hum_3])
    detect_bv_hum = np.concatenate([detect_bv_hum, detect_bv_hum_12, detect_bv_hum_3])

    clcl_color_hum = np.concatenate([clcl_color_hum, cluster_class_hum_12, cluster_class_hum_3])

    age_hum = np.concatenate([age_hum, age_hum_12, age_hum_3])
    ebv_hum = np.concatenate([ebv_hum, ebv_hum_12, ebv_hum_3])



    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_bv_ml_12 = catalog_access.get_hst_color_bv_vega(target=target, classify='ml')
    color_vi_err_ml_12 = catalog_access.get_hst_color_vi_err(target=target, classify='ml')
    color_ub_err_ml_12 = catalog_access.get_hst_color_ub_err(target=target, classify='ml')
    color_bv_err_ml_12 = catalog_access.get_hst_color_bv_err(target=target, classify='ml')
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))
    detect_ub_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0))
    detect_bv_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0))
    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_ml_12 = catalog_access.get_hst_cc_ebv(target=target, classify='ml')

    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_bv_ml_3 = catalog_access.get_hst_color_bv_vega(target=target, classify='ml', cluster_class='class3')
    color_vi_err_ml_3 = catalog_access.get_hst_color_vi_err(target=target, classify='ml', cluster_class='class3')
    color_ub_err_ml_3 = catalog_access.get_hst_color_ub_err(target=target, classify='ml', cluster_class='class3')
    color_bv_err_ml_3 = catalog_access.get_hst_color_bv_err(target=target, classify='ml', cluster_class='class3')
    detect_vi_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F814W') > 0))
    detect_ub_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) > 0))
    detect_bv_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') > 0))
    age_ml_3 = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_ml_3 = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')

    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    color_bv_ml = np.concatenate([color_bv_ml, color_bv_ml_12, color_bv_ml_3])

    color_vi_err_ml = np.concatenate([color_vi_err_ml, color_vi_err_ml_12, color_vi_err_ml_3])
    color_ub_err_ml = np.concatenate([color_ub_err_ml, color_ub_err_ml_12, color_ub_err_ml_3])
    color_bv_err_ml = np.concatenate([color_bv_err_ml, color_bv_err_ml_12, color_bv_err_ml_3])

    detect_vi_ml = np.concatenate([detect_vi_ml, detect_vi_ml_12, detect_vi_ml_3])
    detect_ub_ml = np.concatenate([detect_ub_ml, detect_ub_ml_12, detect_ub_ml_3])
    detect_bv_ml = np.concatenate([detect_bv_ml, detect_bv_ml_12, detect_bv_ml_3])

    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    clcl_qual_color_ml = np.concatenate([clcl_qual_color_ml, cluster_class_qual_ml_12, cluster_class_qual_ml_3])

    age_ml = np.concatenate([age_ml, age_ml_12, age_ml_3])
    ebv_ml = np.concatenate([ebv_ml, ebv_ml_12, ebv_ml_3])

np.save('data_output/color_vi_hum.npy', color_vi_hum)
np.save('data_output/color_ub_hum.npy', color_ub_hum)
np.save('data_output/color_bv_hum.npy', color_bv_hum)
np.save('data_output/color_vi_err_hum.npy', color_vi_err_hum)
np.save('data_output/color_ub_err_hum.npy', color_ub_err_hum)
np.save('data_output/color_bv_err_hum.npy', color_bv_err_hum)
np.save('data_output/detect_vi_hum.npy', detect_vi_hum)
np.save('data_output/detect_ub_hum.npy', detect_ub_hum)
np.save('data_output/detect_bv_hum.npy', detect_bv_hum)
np.save('data_output/clcl_color_hum.npy', clcl_color_hum)
np.save('data_output/age_hum.npy', age_hum)
np.save('data_output/ebv_hum.npy', ebv_hum)
np.save('data_output/color_vi_ml.npy', color_vi_ml)
np.save('data_output/color_ub_ml.npy', color_ub_ml)
np.save('data_output/color_bv_ml.npy', color_bv_ml)
np.save('data_output/color_vi_err_ml.npy', color_vi_err_ml)
np.save('data_output/color_ub_err_ml.npy', color_ub_err_ml)
np.save('data_output/color_bv_err_ml.npy', color_bv_err_ml)
np.save('data_output/detect_vi_ml.npy', detect_vi_ml)
np.save('data_output/detect_ub_ml.npy', detect_ub_ml)
np.save('data_output/detect_bv_ml.npy', detect_bv_ml)
np.save('data_output/clcl_color_ml.npy', clcl_color_ml)
np.save('data_output/clcl_qual_color_ml.npy', clcl_qual_color_ml)
np.save('data_output/age_ml.npy', age_ml)
np.save('data_output/ebv_ml.npy', ebv_ml)



exit()

ubvi_err_hum = np.sqrt(color_ub_err_hum**2 + color_vi_err_hum**2)
bvvi_err_hum = np.sqrt(color_bv_err_hum**2 + color_vi_err_hum**2)
ubvi_err_ml = np.sqrt(color_ub_err_ml**2 + color_vi_err_ml**2)
bvvi_err_ml = np.sqrt(color_bv_err_ml**2 + color_vi_err_ml**2)

noise_cut_ubvi_hum_1 = np.percentile(ubvi_err_hum[(ubvi_err_hum > 0) & (ubvi_err_hum < 1) & (clcl_color_hum == 1)], 95)
noise_cut_ubvi_hum_2 = np.percentile(ubvi_err_hum[(ubvi_err_hum > 0) & (ubvi_err_hum < 1) & (clcl_color_hum == 2)], 95)
noise_cut_ubvi_hum_3 = np.percentile(ubvi_err_hum[(ubvi_err_hum > 0) & (ubvi_err_hum < 1) & (clcl_color_hum == 3)], 95)

noise_cut_ubvi_ml_1 = np.percentile(ubvi_err_ml[(ubvi_err_ml > 0) & (ubvi_err_ml < 1) & (clcl_color_ml == 1)], 95)
noise_cut_ubvi_ml_2 = np.percentile(ubvi_err_ml[(ubvi_err_ml > 0) & (ubvi_err_ml < 1) & (clcl_color_ml == 2)], 95)
noise_cut_ubvi_ml_3 = np.percentile(ubvi_err_ml[(ubvi_err_ml > 0) & (ubvi_err_ml < 1) & (clcl_color_ml == 3)], 95)

noise_cut_bvvi_hum_1 = np.percentile(bvvi_err_hum[(bvvi_err_hum > 0) & (bvvi_err_hum < 1) & (clcl_color_hum == 1)], 95)
noise_cut_bvvi_hum_2 = np.percentile(bvvi_err_hum[(bvvi_err_hum > 0) & (bvvi_err_hum < 1) & (clcl_color_hum == 2)], 95)
noise_cut_bvvi_hum_3 = np.percentile(bvvi_err_hum[(bvvi_err_hum > 0) & (bvvi_err_hum < 1) & (clcl_color_hum == 3)], 95)

noise_cut_bvvi_ml_1 = np.percentile(bvvi_err_ml[(bvvi_err_ml > 0) & (bvvi_err_ml < 1) & (clcl_color_ml == 1)], 95)
noise_cut_bvvi_ml_2 = np.percentile(bvvi_err_ml[(bvvi_err_ml > 0) & (bvvi_err_ml < 1) & (clcl_color_ml == 2)], 95)
noise_cut_bvvi_ml_3 = np.percentile(bvvi_err_ml[(bvvi_err_ml > 0) & (bvvi_err_ml < 1) & (clcl_color_ml == 3)], 95)


print('noise_cut_ubvi_hum_1 ', noise_cut_ubvi_hum_1)
print('noise_cut_ubvi_hum_2 ', noise_cut_ubvi_hum_2)
print('noise_cut_ubvi_hum_3 ', noise_cut_ubvi_hum_3)
print('noise_cut_ubvi_ml_1 ', noise_cut_ubvi_ml_1)
print('noise_cut_ubvi_ml_2 ', noise_cut_ubvi_ml_2)
print('noise_cut_ubvi_ml_3 ', noise_cut_ubvi_ml_3)
print('noise_cut_bvvi_hum_1 ', noise_cut_bvvi_hum_1)
print('noise_cut_bvvi_hum_2 ', noise_cut_bvvi_hum_2)
print('noise_cut_bvvi_hum_3 ', noise_cut_bvvi_hum_3)
print('noise_cut_bvvi_ml_1 ', noise_cut_bvvi_ml_1)
print('noise_cut_bvvi_ml_2 ', noise_cut_bvvi_ml_2)
print('noise_cut_bvvi_ml_3 ', noise_cut_bvvi_ml_3)

#
# err_bins = np.linspace(0, 1, 60)
# center_of_err_bins = (err_bins[:-1] + err_bins[1:]) / 2
# hist_err_ubvi_hum_1 = np.histogram(ubvi_err_hum[clcl_color_hum == 1], bins=err_bins)[0]
# hist_err_ubvi_hum_2 = np.histogram(ubvi_err_hum[clcl_color_hum == 2], bins=err_bins)[0]
# hist_err_ubvi_hum_3 = np.histogram(ubvi_err_hum[clcl_color_hum == 3], bins=err_bins)[0]
#
# hist_err_ubvi_ml_1 = np.histogram(ubvi_err_ml[clcl_color_ml == 1], bins=err_bins)[0]
# hist_err_ubvi_ml_2 = np.histogram(ubvi_err_ml[clcl_color_ml == 2], bins=err_bins)[0]
# hist_err_ubvi_ml_3 = np.histogram(ubvi_err_ml[clcl_color_ml == 3], bins=err_bins)[0]
#
# plt.step(center_of_err_bins, hist_err_ubvi_hum_1, where='mid')
# plt.step(center_of_err_bins, hist_err_ubvi_hum_2, where='mid')
# plt.step(center_of_err_bins, hist_err_ubvi_hum_3, where='mid')
# plt.plot([noise_cut_ubvi_hum, noise_cut_ubvi_hum], [0, 800])
# plt.show()
#
# plt.step(center_of_err_bins, hist_err_ubvi_ml_1, where='mid')
# plt.step(center_of_err_bins, hist_err_ubvi_ml_2, where='mid')
# plt.step(center_of_err_bins, hist_err_ubvi_ml_3, where='mid')
# plt.plot([noise_cut_ubvi_ml, noise_cut_ubvi_ml], [0, 800])
# plt.show()
#
# exit()


mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3


x_lim_ubvi = (-0.6, 1.7)
y_lim_ubvi = (0.9, -2.2)

x_lim_bvvi = (-0.6, 1.7)
y_lim_bvvi = (1.5, -0.7)

# bins
x_bins_gauss_ubvi = np.linspace(x_lim_ubvi[0], x_lim_ubvi[1], 100)
y_bins_gauss_ubvi = np.linspace(y_lim_ubvi[1], y_lim_ubvi[0], 100)
# get a mesh
x_mesh_ubvi, y_mesh_ubvi = np.meshgrid(x_bins_gauss_ubvi, y_bins_gauss_ubvi)
gauss_map_ubvi_hum_1 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
gauss_map_ubvi_ml_1 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
gauss_map_ubvi_hum_2 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
gauss_map_ubvi_ml_2 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
gauss_map_ubvi_hum_12 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
gauss_map_ubvi_ml_12 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
gauss_map_ubvi_hum_3 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
gauss_map_ubvi_ml_3 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))

noise_gauss_map_ubvi_hum_1 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
noise_gauss_map_ubvi_ml_1 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
noise_gauss_map_ubvi_hum_2 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
noise_gauss_map_ubvi_ml_2 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
noise_gauss_map_ubvi_hum_12 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
noise_gauss_map_ubvi_ml_12 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
noise_gauss_map_ubvi_hum_3 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))
noise_gauss_map_ubvi_ml_3 = np.zeros((len(x_bins_gauss_ubvi), len(y_bins_gauss_ubvi)))

for color_index in range(len(color_vi_hum)):
    gauss = gauss2d(x=x_mesh_ubvi, y=y_mesh_ubvi, x0=color_vi_hum[color_index], y0=color_ub_hum[color_index],
                    sig_x=color_vi_err_hum[color_index], sig_y=color_ub_err_hum[color_index])
    if mask_class_1_hum[color_index]:
        gauss_map_ubvi_hum_1 += gauss
        gauss_map_ubvi_hum_12 += gauss
        if ubvi_err_hum[color_index] > noise_cut_ubvi_hum_1:
            noise_gauss_map_ubvi_hum_1 += gauss
            noise_gauss_map_ubvi_hum_12 += gauss
    if mask_class_2_hum[color_index]:
        gauss_map_ubvi_hum_2 += gauss
        gauss_map_ubvi_hum_12 += gauss
        if ubvi_err_hum[color_index] > noise_cut_ubvi_hum_2:
            noise_gauss_map_ubvi_hum_2 += gauss
            noise_gauss_map_ubvi_hum_12 += gauss
    if mask_class_3_hum[color_index]:
        gauss_map_ubvi_hum_3 += gauss
        if ubvi_err_hum[color_index] > noise_cut_ubvi_hum_3:
            noise_gauss_map_ubvi_hum_3 += gauss

for color_index in range(len(color_vi_ml)):
    gauss = gauss2d(x=x_mesh_ubvi, y=y_mesh_ubvi, x0=color_vi_ml[color_index], y0=color_ub_ml[color_index],
                    sig_x=color_vi_err_ml[color_index], sig_y=color_ub_err_ml[color_index])
    if mask_class_1_ml[color_index]:
        gauss_map_ubvi_ml_1 += gauss
        gauss_map_ubvi_ml_12 += gauss
        if ubvi_err_ml[color_index] > noise_cut_ubvi_ml_1:
            noise_gauss_map_ubvi_ml_1 += gauss
            noise_gauss_map_ubvi_ml_12 += gauss
    if mask_class_2_ml[color_index]:
        gauss_map_ubvi_ml_2 += gauss
        gauss_map_ubvi_ml_12 += gauss
        if ubvi_err_ml[color_index] > noise_cut_ubvi_ml_2:
            noise_gauss_map_ubvi_ml_2 += gauss
            noise_gauss_map_ubvi_ml_12 += gauss
    if mask_class_3_ml[color_index]:
        gauss_map_ubvi_ml_3 += gauss
        if ubvi_err_ml[color_index] > noise_cut_ubvi_ml_3:
            noise_gauss_map_ubvi_ml_3 += gauss


# bins
x_bins_gauss_bvvi = np.linspace(x_lim_bvvi[0], x_lim_bvvi[1], 100)
y_bins_gauss_bvvi = np.linspace(y_lim_bvvi[1], y_lim_bvvi[0], 100)
# get a mesh
x_mesh_bvvi, y_mesh_bvvi = np.meshgrid(x_bins_gauss_bvvi, y_bins_gauss_bvvi)
gauss_map_bvvi_hum_1 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
gauss_map_bvvi_ml_1 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
gauss_map_bvvi_hum_2 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
gauss_map_bvvi_ml_2 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
gauss_map_bvvi_hum_12 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
gauss_map_bvvi_ml_12 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
gauss_map_bvvi_hum_3 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
gauss_map_bvvi_ml_3 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))

noise_gauss_map_bvvi_hum_1 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
noise_gauss_map_bvvi_ml_1 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
noise_gauss_map_bvvi_hum_2 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
noise_gauss_map_bvvi_ml_2 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
noise_gauss_map_bvvi_hum_12 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
noise_gauss_map_bvvi_ml_12 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
noise_gauss_map_bvvi_hum_3 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))
noise_gauss_map_bvvi_ml_3 = np.zeros((len(x_bins_gauss_bvvi), len(y_bins_gauss_bvvi)))

for color_index in range(len(color_vi_hum)):
    gauss = gauss2d(x=x_mesh_bvvi, y=y_mesh_bvvi, x0=color_vi_hum[color_index], y0=color_bv_hum[color_index],
                    sig_x=color_vi_err_hum[color_index], sig_y=color_bv_err_hum[color_index])
    if mask_class_1_hum[color_index]:
        gauss_map_bvvi_hum_1 += gauss
        gauss_map_bvvi_hum_12 += gauss
        if bvvi_err_hum[color_index] > noise_cut_bvvi_hum_1:
            noise_gauss_map_bvvi_hum_1 += gauss
            noise_gauss_map_bvvi_hum_12 += gauss
    if mask_class_2_hum[color_index]:
        gauss_map_bvvi_hum_2 += gauss
        gauss_map_bvvi_hum_12 += gauss
        if bvvi_err_hum[color_index] > noise_cut_bvvi_hum_2:
            noise_gauss_map_bvvi_hum_2 += gauss
            noise_gauss_map_bvvi_hum_12 += gauss
    if mask_class_3_hum[color_index]:
        gauss_map_bvvi_hum_3 += gauss
        if bvvi_err_hum[color_index] > noise_cut_bvvi_hum_3:
            noise_gauss_map_bvvi_hum_3 += gauss

for color_index in range(len(color_vi_ml)):
    gauss = gauss2d(x=x_mesh_bvvi, y=y_mesh_bvvi, x0=color_vi_ml[color_index], y0=color_bv_ml[color_index],
                    sig_x=color_vi_err_ml[color_index], sig_y=color_bv_err_ml[color_index])
    if mask_class_1_ml[color_index]:
        gauss_map_bvvi_ml_1 += gauss
        gauss_map_bvvi_ml_12 += gauss
        if bvvi_err_ml[color_index] > noise_cut_bvvi_ml_1:
            noise_gauss_map_bvvi_ml_1 += gauss
            noise_gauss_map_bvvi_ml_12 += gauss
    if mask_class_2_ml[color_index]:
        gauss_map_bvvi_ml_2 += gauss
        gauss_map_bvvi_ml_12 += gauss
        if bvvi_err_ml[color_index] > noise_cut_bvvi_ml_2:
            noise_gauss_map_bvvi_ml_2 += gauss
            noise_gauss_map_bvvi_ml_12 += gauss
    if mask_class_3_ml[color_index]:
        gauss_map_bvvi_ml_3 += gauss
        if bvvi_err_ml[color_index] > noise_cut_bvvi_ml_3:
            noise_gauss_map_bvvi_ml_3 += gauss


np.save('data_output/x_lim_ubvi.npy', x_lim_ubvi)
np.save('data_output/y_lim_ubvi.npy', y_lim_ubvi)
np.save('data_output/x_lim_bvvi.npy', x_lim_bvvi)
np.save('data_output/y_lim_bvvi.npy', y_lim_bvvi)
np.save('data_output/x_bins_gauss_ubvi.npy', x_bins_gauss_ubvi)
np.save('data_output/y_bins_gauss_ubvi.npy', y_bins_gauss_ubvi)
np.save('data_output/x_bins_gauss_bvvi.npy', x_bins_gauss_bvvi)
np.save('data_output/y_bins_gauss_bvvi.npy', y_bins_gauss_bvvi)


np.save('data_output/gauss_map_ubvi_hum_1.npy', gauss_map_ubvi_hum_1)
np.save('data_output/gauss_map_ubvi_ml_1.npy', gauss_map_ubvi_ml_1)
np.save('data_output/gauss_map_ubvi_hum_2.npy', gauss_map_ubvi_hum_2)
np.save('data_output/gauss_map_ubvi_ml_2.npy', gauss_map_ubvi_ml_2)
np.save('data_output/gauss_map_ubvi_hum_12.npy', gauss_map_ubvi_hum_12)
np.save('data_output/gauss_map_ubvi_ml_12.npy', gauss_map_ubvi_ml_12)
np.save('data_output/gauss_map_ubvi_hum_3.npy', gauss_map_ubvi_hum_3)
np.save('data_output/gauss_map_ubvi_ml_3.npy', gauss_map_ubvi_ml_3)
np.save('data_output/gauss_map_bvvi_hum_1.npy', gauss_map_bvvi_hum_1)
np.save('data_output/gauss_map_bvvi_ml_1.npy', gauss_map_bvvi_ml_1)
np.save('data_output/gauss_map_bvvi_hum_2.npy', gauss_map_bvvi_hum_2)
np.save('data_output/gauss_map_bvvi_ml_2.npy', gauss_map_bvvi_ml_2)
np.save('data_output/gauss_map_bvvi_hum_12.npy', gauss_map_bvvi_hum_12)
np.save('data_output/gauss_map_bvvi_ml_12.npy', gauss_map_bvvi_ml_12)
np.save('data_output/gauss_map_bvvi_hum_3.npy', gauss_map_bvvi_hum_3)
np.save('data_output/gauss_map_bvvi_ml_3.npy', gauss_map_bvvi_ml_3)


np.save('data_output/noise_gauss_map_ubvi_hum_1.npy', noise_gauss_map_ubvi_hum_1)
np.save('data_output/noise_gauss_map_ubvi_ml_1.npy', noise_gauss_map_ubvi_ml_1)
np.save('data_output/noise_gauss_map_ubvi_hum_2.npy', noise_gauss_map_ubvi_hum_2)
np.save('data_output/noise_gauss_map_ubvi_ml_2.npy', noise_gauss_map_ubvi_ml_2)
np.save('data_output/noise_gauss_map_ubvi_hum_12.npy', noise_gauss_map_ubvi_hum_12)
np.save('data_output/noise_gauss_map_ubvi_ml_12.npy', noise_gauss_map_ubvi_ml_12)
np.save('data_output/noise_gauss_map_ubvi_hum_3.npy', noise_gauss_map_ubvi_hum_3)
np.save('data_output/noise_gauss_map_ubvi_ml_3.npy', noise_gauss_map_ubvi_ml_3)
np.save('data_output/noise_gauss_map_bvvi_hum_1.npy', noise_gauss_map_bvvi_hum_1)
np.save('data_output/noise_gauss_map_bvvi_ml_1.npy', noise_gauss_map_bvvi_ml_1)
np.save('data_output/noise_gauss_map_bvvi_hum_2.npy', noise_gauss_map_bvvi_hum_2)
np.save('data_output/noise_gauss_map_bvvi_ml_2.npy', noise_gauss_map_bvvi_ml_2)
np.save('data_output/noise_gauss_map_bvvi_hum_12.npy', noise_gauss_map_bvvi_hum_12)
np.save('data_output/noise_gauss_map_bvvi_ml_12.npy', noise_gauss_map_bvvi_ml_12)
np.save('data_output/noise_gauss_map_bvvi_hum_3.npy', noise_gauss_map_bvvi_hum_3)
np.save('data_output/noise_gauss_map_bvvi_ml_3.npy', noise_gauss_map_bvvi_ml_3)


fig, ax = plt.subplots(2, 2, figsize=(15, 15))
fontsize = 17

ax[0, 0].imshow(gauss_map_ubvi_hum_12, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]),
                cmap='Greys', vmin=0, vmax=np.max(gauss_map_ubvi_hum_12))

ax[0, 1].imshow(gauss_map_ubvi_ml_12, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]),
                cmap='Greys', vmin=0, vmax=np.max(gauss_map_ubvi_ml_12))

ax[1, 0].imshow(gauss_map_bvvi_hum_12, origin='lower', extent=(x_lim_bvvi[0], x_lim_bvvi[1], y_lim_bvvi[1], y_lim_bvvi[0]),
                cmap='Greys', vmin=0, vmax=np.max(gauss_map_bvvi_hum_12))

ax[1, 1].imshow(gauss_map_bvvi_ml_12, origin='lower', extent=(x_lim_bvvi[0], x_lim_bvvi[1], y_lim_bvvi[1], y_lim_bvvi[0]),
                cmap='Greys', vmin=0, vmax=np.max(gauss_map_bvvi_ml_12))

ax[0, 0].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='--', label=r'BC03, Z$_{\odot}$')
ax[0, 0].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='--', label=r'BC03, Z$_{\odot}/50$')

ax[0, 1].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='--')
ax[0, 1].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='--')

ax[1, 0].plot(model_vi_sol, model_bv_sol, color='darkred', linewidth=2, linestyle='--')
ax[1, 0].plot(model_vi_sol50, model_bv_sol50, color='darkorange', linewidth=2, linestyle='--')

ax[1, 1].plot(model_vi_sol, model_bv_sol, color='darkred', linewidth=2, linestyle='--')
ax[1, 1].plot(model_vi_sol50, model_bv_sol50, color='darkorange', linewidth=2, linestyle='--')



ax[0, 0].set_title('Hum Class 1|2', fontsize=fontsize)
ax[0, 1].set_title('ML Class 1|2', fontsize=fontsize)


ax[0, 0].set_xlim(x_lim_ubvi)
ax[0, 1].set_xlim(x_lim_ubvi)
ax[1, 0].set_xlim(x_lim_bvvi)
ax[1, 1].set_xlim(x_lim_bvvi)

ax[0, 0].set_ylim(y_lim_ubvi)
ax[0, 1].set_ylim(y_lim_ubvi)
ax[1, 0].set_ylim(y_lim_bvvi)
ax[1, 1].set_ylim(y_lim_bvvi)

ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])

ax[0, 1].set_yticklabels([])
ax[1, 1].set_yticklabels([])

ax[0, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 0].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)
ax[1, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


ax[0, 0].legend(frameon=False, loc=3, fontsize=fontsize)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


# plt.show()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/regions_overview.png')
exit()

# # kernel_std = 1.5
# # kernel = Gaussian2DKernel(x_stddev=kernel_std)
# # gauss_map = convolve(gauss_map, kernel)
# bkg_estimator = MedianBackground()
# bkg = Background2D(gauss_map, (20, 20), filter_size=(3, 3),
#                    bkg_estimator=bkg_estimator)
# # gauss_map -= bkg.background  # subtract the background
# threshold = 2.5 * bkg.background_rms
# segment_map = detect_sources(gauss_map, threshold, npixels=10)
#
# norm = ImageNormalize(stretch=SqrtStretch())
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
# ax1.imshow(gauss_map, origin='lower', cmap='Greys_r', norm=norm)
# ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
#            interpolation='nearest')

np.save('gauss_map.npy', gauss_map)
gauss_map = np.load('gauss_map.npy')




exit()


bkg_estimator = MedianBackground()
bkg = Background2D(gauss_map, (50, 50), filter_size=(3, 3),
                   bkg_estimator=bkg_estimator)
gauss_map -= bkg.background  # subtract the background
threshold = 1.5 * bkg.background_rms

kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
convolved_data = convolve(gauss_map, kernel)

segment_map = detect_sources(convolved_data, threshold, npixels=10)


from photutils.segmentation import deblend_sources

segm_deblend = deblend_sources(convolved_data, segment_map,
                               npixels=10, nlevels=32, contrast=0.001,
                               progress_bar=False)



norm = ImageNormalize(stretch=SqrtStretch())
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12.5))
ax1.imshow(gauss_map, origin='lower', cmap='Greys_r', norm=norm)
ax1.set_title('Background-subtracted Data')
ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
           interpolation='nearest')
ax2.set_title('Segmentation Image')

ax3.imshow(segm_deblend)
plt.show()









exit()


x_lim = (-0.6, 2.1)
y_lim = (0.9, -2.2)

# bins
x_bins_gauss = np.linspace(x_lim[0], x_lim[1], 100)
y_bins_gauss = np.linspace(y_lim[1], y_lim[0], 100)

from scipy.interpolate import griddata

import lmfit
from lmfit.lineshapes import gaussian2d, lorentzian



def lorentzian2d(x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1.,
                 rotation=0):
    """Return a two dimensional lorentzian.

    The maximum of the peak occurs at ``centerx`` and ``centery``
    with widths ``sigmax`` and ``sigmay`` in the x and y directions
    respectively. The peak can be rotated by choosing the value of ``rotation``
    in radians.
    """
    xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
    yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
    R = (xp/sigmax)**2 + (yp/sigmay)**2

    return 2*amplitude*lorentzian(R)/(np.pi*sigmax*sigmay)


def gauss2d_rot(xdata, amp, x0, y0, sig_x, sig_y, theta):
    x = xdata[0]
    y = xdata[1]
    sigx2 = sig_x ** 2
    sigy2 = sig_y ** 2
    a = np.cos(theta) ** 2 / (2 * sigx2) + np.sin(theta) ** 2 / (2 * sigy2)
    b = np.sin(theta) ** 2 / (2 * sigx2) + np.cos(theta) ** 2 / (2 * sigy2)
    c = np.sin(2 * theta) / (4 * sigx2) - np.sin(2 * theta) / (4 * sigy2)

    expo = -a * (x - x0) ** 2 - b * (y - y0) ** 2 - 2 * c * (x - x0) * (y - y0)

    return amp * np.exp(expo)


# npoints = 10000
# x = np.random.rand(npoints)*10 - 4
# y = np.random.rand(npoints)*5 - 3
# z = lorentzian2d(x, y, amplitude=30, centerx=2, centery=-.5, sigmax=.6,
#                  sigmay=1.2, rotation=30*np.pi/180)
# z += 2*(np.random.rand(*z.shape)-.5)
# error = np.sqrt(z+1)
#
# X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
#                    np.linspace(y.min(), y.max(), 100))
#

Z = griddata((x, y), z, (X, Y), method='linear', fill_value=0)

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
art = ax.pcolor(X, Y, Z, shading='auto')
plt.colorbar(art, ax=ax, label='z')
plt.show()


model = lmfit.Model(lorentzian2d, independent_vars=['x', 'y'])
params = model.make_params(amplitude=10, centerx=x[np.argmax(z)],
                           centery=y[np.argmax(z)])
params['rotation'].set(value=.1, min=0, max=np.pi/2)
params['sigmax'].set(value=1, min=0)
params['sigmay'].set(value=2, min=0)

result = model.fit(z, x=x, y=y, params=params, weights=None)
print(lmfit.report_fit(result))



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

vmax = np.nanpercentile(Z, 99.9)

ax = axs[0, 0]
art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
plt.colorbar(art, ax=ax, label='z')
ax.set_title('Data')

ax = axs[0, 1]
fit = model.func(X, Y, **result.best_values)
art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
plt.colorbar(art, ax=ax, label='z')
ax.set_title('Fit')

ax = axs[1, 0]
fit = model.func(X, Y, **result.best_values)
art = ax.pcolor(X, Y, Z-fit, vmin=0, vmax=10, shading='auto')
plt.colorbar(art, ax=ax, label='z')
ax.set_title('Data - Fit')

for ax in axs.ravel():
    ax.set_xlabel('x')
    ax.set_ylabel('y')
axs[1, 1].remove()
plt.show()



exit()

import scipy.optimize as opt

initial_guess = (np.max(gauss_map) * 0.5, 0.6, 0.5, 0.2, 0.2, np.pi/2)


popt, pcov = opt.curve_fit(f=gauss2d_rot, xdata=np.array([x_mesh, y_mesh]), ydata=gauss_map, p0=initial_guess)

print(popt)
print(pcov)

exit()

gmodel = Model(gauss2d_rot)

params = Parameters()
params.add('amp', value=np.max(gauss_map) * 0.5, vary=True, min=0, max=np.max(gauss_map)*1.5)
params.add('x0', value=0.6, vary=True, min=-0.5, max=1.5)
params.add('y0', value=-0.5, vary=True, min=-1.7, max=0.5)
params.add('sig_x', value=0.2, vary=True, min=0.01, max=0.5)
params.add('sig_Y', value=0.2, vary=True, min=0.01, max=0.5)
params.add('theta', value=np.pi/2, vary=True, min=0, max=np.pi)

result = gmodel.fit(gauss_map, data=np.array([x_mesh, y_mesh]), params=params)

print(result.fit_report())

# gauss2d_rot(x, y, amp, x0, y0, sig_x, sig_y, theta)





plt.imshow(gauss_map, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
             cmap='Greys', vmin=10, vmax=np.max(gauss_map))

plt.show()


exit()






figure = plt.figure(figsize=(30, 10))
fontsize = 26

ax_cc_sn_vi = figure.add_axes([0.05, 0.05, 0.23, 0.9])
ax_cc_sn_ub = figure.add_axes([0.28, 0.05, 0.23, 0.9])
ax_cc_reg = figure.add_axes([0.51, 0.05, 0.23, 0.9])
ax_cc_ebv = figure.add_axes([0.74, 0.05, 0.23, 0.9])

ax_cbar_sn_vi = figure.add_axes([0.07, 0.96, 0.19, 0.015])
ax_cbar_sn_ub = figure.add_axes([0.30, 0.96, 0.19, 0.015])
ax_cbar_ebv = figure.add_axes([0.76, 0.96, 0.19, 0.015])


ax_cc_sn_vi.plot([], [], color='white', label='N=%i' % (sum((mask_class_1_ml + mask_class_2_ml))))
ax_cc_sn_vi.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}$')
ax_cc_sn_vi.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}/50$')

ax_cc_sn_ub.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='--')
ax_cc_sn_ub.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

ax_cc_reg.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='--')
ax_cc_reg.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

ax_cc_ebv.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='--')
ax_cc_ebv.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')


cmap_vi = matplotlib.cm.get_cmap('viridis')
norm_vi = matplotlib.colors.Normalize(vmin=0, vmax=0.12)

cmap_ub = matplotlib.cm.get_cmap('plasma')
norm_ub = matplotlib.colors.Normalize(vmin=0, vmax=0.5)

cmap_ebv = matplotlib.cm.get_cmap('cividis')
norm_ebv = matplotlib.colors.Normalize(vmin=0, vmax=0.5)


ax_cc_sn_vi.imshow(snr_vi_map.T, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]), cmap=cmap_vi, norm=norm_vi)
ax_cc_sn_ub.imshow(snr_ub_map.T, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]), cmap=cmap_ub, norm=norm_ub)


ax_cc_ebv.imshow(ebv_map.T, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]), cmap=cmap_ebv, norm=norm_ebv)



ColorbarBase(ax_cbar_sn_vi, orientation='horizontal', cmap=cmap_vi, norm=norm_vi, extend='neither', ticks=None)
ax_cbar_sn_vi.set_xlabel(r'$\sigma_{\rm V-I}$', labelpad=-4, fontsize=fontsize)
ax_cbar_sn_vi.tick_params(axis='both', which='both', width=2, direction='in', top=False, bottom=True, labelbottom=True, labeltop=False, labelsize=fontsize)

ColorbarBase(ax_cbar_sn_ub, orientation='horizontal', cmap=cmap_ub, norm=norm_ub, extend='neither', ticks=None)
ax_cbar_sn_ub.set_xlabel(r'$\sigma_{\rm U-B}$', labelpad=-4, fontsize=fontsize)
ax_cbar_sn_ub.tick_params(axis='both', which='both', width=2, direction='in', top=False, bottom=True, labelbottom=True, labeltop=False, labelsize=fontsize)

ColorbarBase(ax_cbar_ebv, orientation='horizontal', cmap=cmap_ebv, norm=norm_ebv, extend='neither', ticks=None)
ax_cbar_ebv.set_xlabel(r'E(B-V) [mag]', labelpad=-4, fontsize=fontsize)
ax_cbar_ebv.tick_params(axis='both', which='both', width=2, direction='in', top=False, bottom=True, labelbottom=True, labeltop=False, labelsize=fontsize)


ax_cc_sn_vi.set_xlim(x_lim)
ax_cc_sn_ub.set_xlim(x_lim)
ax_cc_reg.set_xlim(x_lim)
ax_cc_ebv.set_xlim(x_lim)
ax_cc_sn_vi.set_ylim(y_lim)
ax_cc_sn_ub.set_ylim(y_lim)
ax_cc_reg.set_ylim(y_lim)
ax_cc_ebv.set_ylim(y_lim)



# young clusters
ellipse = Ellipse(xy=(0.3, -1.35), width=0.3, height=0.9, angle=-60,
                        edgecolor='darkblue', fc='None', lw=5)
ax_cc_reg.add_patch(ellipse)
ax_cc_reg.text(0.3, -1.35, '1)', horizontalalignment='center', verticalalignment='bottom', color='darkblue', fontsize=fontsize+10)

# intermediate age
ellipse = Ellipse(xy=(0.65, -0.4), width=0.4, height=1.1, angle=-5,
                        edgecolor='tab:green', fc='None', lw=5)
ax_cc_reg.add_patch(ellipse)
ax_cc_reg.text(0.65, -0.4, '2)', horizontalalignment='center', verticalalignment='bottom', color='tab:green', fontsize=fontsize+10)


# globular clusters
ellipse = Ellipse(xy=(1.1, -0.1), width=0.3, height=0.5, angle=0,
                        edgecolor='crimson', fc='None', lw=5)
ax_cc_reg.add_patch(ellipse)
ax_cc_reg.text(1.1, -0.1, '3)', horizontalalignment='center', verticalalignment='bottom', color='crimson', fontsize=fontsize+10)



# add reddening vector
# intrinsic colors
vi_int = 0.5
ub_int = -1.9
max_av = 2.2
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)
max_color_ext_vi_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av+0.1)
max_color_ext_ub_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av+0.1)
x_av = np.linspace(vi_int, vi_int + max_color_ext_vi, 100)
y_av = np.linspace(ub_int, ub_int + max_color_ext_ub, 100)
av_value = np.linspace(0.0, max_av, 100)

cmap_av = matplotlib.cm.get_cmap('Spectral_r')
norm_av = matplotlib.colors.Normalize(vmin=av_value[:-1].min(), vmax=av_value[:-1].max())

points = np.array([x_av[:-1], y_av[:-1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=cmap_av, norm=norm_av, alpha=1.0, zorder=9)

lc.set_array(av_value[:-1])
lc.set_linewidth(20)
line = ax_cc_reg.add_collection(lc)

ax_cc_reg.annotate("", xy=(vi_int+max_color_ext_vi_arr, ub_int+max_color_ext_ub_arr), xytext=(x_av[-1], y_av[-1]),
               arrowprops=dict(arrowstyle="-|>, head_width=2.5, head_length=2.5", linewidth=1,
                               color=cmap_av(norm_av(av_value[-1]))), zorder=10)


ax_cc_reg.text(0.90, -1.67, r'A$_{\rm V}$', fontsize=fontsize+3, rotation=-30)
ax_cc_reg.text(0.48, -1.93, r'0', fontsize=fontsize+3, rotation=-30)
ax_cc_reg.text(1.32, -1.40, r'2', fontsize=fontsize+3, rotation=-30)


# ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap_av, norm=norm_av, extend='neither', ticks=[0, 0.5, 1, 1.5, 2.0])
# ax_cbar.set_xlabel(r'A$_{\rm V}$ [mag]', labelpad=4, fontsize=fontsize)
# ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
#                     labeltop=True, labelsize=fontsize)
#


# ax_cc_reg.set_title('Class 1|2 ML', fontsize=fontsize)

ax_cc_sn_ub.set_yticklabels([])
ax_cc_reg.set_yticklabels([])
ax_cc_ebv.set_yticklabels([])

ax_cc_sn_vi.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_sn_ub.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_reg.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_ebv.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_sn_vi.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_cc_reg.legend(frameon=False, loc=3, fontsize=fontsize)
ax_cc_sn_vi.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_ub.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_reg.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_ebv.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

# fig.subplots_adjust(wspace=0, hspace=0)
figure.savefig('plot_output/color_color_regions.png')
figure.savefig('plot_output/color_color_regions.pdf')
