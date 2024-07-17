import numpy as np
import matplotlib.pyplot as plt
from photometry_tools import helper_func, data_access
from scipy.spatial import ConvexHull
from astropy.io import fits
from matplotlib import patheffects
import dust_tools

age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_nuvb_sol = np.load('../color_color/data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')



cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                           hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                           morph_mask_path=morph_mask_path,
                                           sample_table_path=sample_table_path)



cb19_file_name = '/home/benutzer/data/PHANGS_products/models/cb19/cb2019_z020_chab_hr_xmilesi_ssp.fits'

cb19_table = fits.open(cb19_file_name)
time_steps = cb19_table[5].data
photo_values = cb19_table[3].data

model_cb19_sol_nuv_flux = helper_func.conv_ab_mag2mjy(photo_values['UVIS1_f275w_AB'])
model_cb19_sol_u_flux = helper_func.conv_ab_mag2mjy(photo_values['UVIS1_f336w_AB'])
model_cb19_sol_b_flux = helper_func.conv_ab_mag2mjy(photo_values['UVIS1_f438w_AB'])
model_cb19_sol_v_flux = helper_func.conv_ab_mag2mjy(photo_values['UVIS1_f555w_AB'])
model_cb19_sol_i_flux = helper_func.conv_ab_mag2mjy(photo_values['UVIS1_f814w_AB'])

model_cb19_sol_nuv_vega = helper_func.conv_mjy2vega(flux=model_cb19_sol_nuv_flux, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'), vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
model_cb19_sol_u_vega = helper_func.conv_mjy2vega(flux=model_cb19_sol_u_flux, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'), vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
model_cb19_sol_b_vega = helper_func.conv_mjy2vega(flux=model_cb19_sol_b_flux, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'), vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_cb19_sol_v_vega = helper_func.conv_mjy2vega(flux=model_cb19_sol_v_flux, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'), vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
model_cb19_sol_i_vega = helper_func.conv_mjy2vega(flux=model_cb19_sol_i_flux, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'), vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))

model_nuvb_cb19_sol = model_cb19_sol_nuv_vega - model_cb19_sol_b_vega
model_ub_cb19_sol = model_cb19_sol_u_vega - model_cb19_sol_b_vega
model_bv_cb19_sol = model_cb19_sol_b_vega - model_cb19_sol_v_vega
model_vi_cb19_sol = model_cb19_sol_v_vega - model_cb19_sol_i_vega
age_mod_cb19_sol = time_steps['age-yr'] * 1e-6


# get model
hdu_a_sol_neb = fits.open('../cigale_model/sfh2exp/no_dust/sol_met_gas/out/models-block-0.fits')
data_mod_sol_neb = hdu_a_sol_neb[1].data


ew_h_alpha = data_mod_sol_neb['param.EW(656.3/1.0)']
age_mod_sol_neb = data_mod_sol_neb['sfh.age']
logu_mod_sol_neb = data_mod_sol_neb['nebular.logU']
ne_mod_sol_neb = data_mod_sol_neb['nebular.ne']
f_esc_mod_sol_neb = data_mod_sol_neb['nebular.f_esc']

flux_f275w_sol_neb = data_mod_sol_neb['F275W_UVIS_CHIP2']
flux_f336w_sol_neb = data_mod_sol_neb['F336W_UVIS_CHIP2']
flux_f438w_sol_neb = data_mod_sol_neb['F438W_UVIS_CHIP2']
flux_f555w_sol_neb = data_mod_sol_neb['F555W_UVIS_CHIP2']
flux_f814w_sol_neb = data_mod_sol_neb['F814W_UVIS_CHIP2']



nuv_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F275W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4




color_vi_hum = np.load('../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../color_color/data_output/color_ub_hum.npy')
color_nuvb_hum = np.load('../color_color/data_output/color_nuvb_hum.npy')
color_bv_hum = np.load('../color_color/data_output/color_bv_hum.npy')
color_vi_err_hum = np.load('../color_color/data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('../color_color/data_output/color_ub_err_hum.npy')
color_nuvb_err_hum = np.load('../color_color/data_output/color_nuvb_err_hum.npy')
color_bv_err_hum = np.load('../color_color/data_output/color_bv_err_hum.npy')
detect_nuv_hum = np.load('../color_color/data_output/detect_nuv_hum.npy')
detect_u_hum = np.load('../color_color/data_output/detect_u_hum.npy')
detect_b_hum = np.load('../color_color/data_output/detect_b_hum.npy')
detect_v_hum = np.load('../color_color/data_output/detect_v_hum.npy')
detect_i_hum = np.load('../color_color/data_output/detect_i_hum.npy')
clcl_color_hum = np.load('../color_color/data_output/clcl_color_hum.npy')
age_hum = np.load('../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../color_color/data_output/ebv_hum.npy')


# color range limitations
# x_lim_vi = (-0.6, 1.9)
# # y_lim_ub = (0.9, -2.2)
#
# y_lim_nuvb = (3.2, -2.9)
# y_lim_ub = (2.1, -2.2)
# y_lim_bv = (1.9, -0.7)
#

x_lim_vi = (-0.7, 3.8)

y_lim_nuvb = (3.2, -2.9)
y_lim_ub = (2.1, -2.2)
y_lim_bv = (1.9, -0.7)

for ebv in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.8, 2.0, 2.2]:


    av_val = ebv * 3.1

    color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=av_val)
    color_ext_nuvb = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=nuv_wave, wave2=b_wave, av=av_val)
    color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=av_val)
    color_ext_bv = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=b_wave, wave2=v_wave, av=av_val)


    mag_nuv_sol_neb = helper_func.conv_mjy2vega(flux=flux_f275w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                                 vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
    mag_u_sol_neb = helper_func.conv_mjy2vega(flux=flux_f336w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                                 vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
    mag_b_sol_neb = helper_func.conv_mjy2vega(flux=flux_f438w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                                 vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
    mag_v_sol_neb = helper_func.conv_mjy2vega(flux=flux_f555w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                                 vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
    mag_i_sol_neb = helper_func.conv_mjy2vega(flux=flux_f814w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                                 vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
    model_nuvb_sol_neb = mag_nuv_sol_neb - mag_b_sol_neb + color_ext_nuvb
    model_ub_sol_neb = mag_u_sol_neb - mag_b_sol_neb + color_ext_ub
    model_bv_sol_neb = mag_b_sol_neb - mag_v_sol_neb + color_ext_bv
    model_vi_sol_neb = mag_v_sol_neb - mag_i_sol_neb + color_ext_vi





    mask_class_12_hum = (clcl_color_hum == 1) | (clcl_color_hum == 2)

    mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
    mask_detect_bvi_hum = detect_b_hum * detect_v_hum * detect_i_hum
    mask_detect_nuvbvi_hum = detect_nuv_hum * detect_b_hum * detect_v_hum * detect_i_hum

    mask_good_colors_nuvbvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                                 (color_nuvb_hum > (y_lim_nuvb[1] - 1)) & (color_nuvb_hum < (y_lim_nuvb[0] + 1)) &
                                 mask_detect_nuvbvi_hum)
    mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                                 (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                                 mask_detect_ubvi_hum)

    mask_good_colors_bvvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                                 (color_bv_hum > (y_lim_bv[1] - 1)) & (color_bv_hum < (y_lim_bv[0] + 1)) &
                                 mask_detect_bvi_hum)

    # get gauss und segmentations
    n_bins = 90
    threshold_fact = 3
    kernal_std = 1.0
    contrast = 0.01

    gauss_dict_nuvbvi_hum_12 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_12_hum * mask_good_colors_nuvbvi_hum],
                                                  y_data=color_nuvb_hum[mask_class_12_hum * mask_good_colors_nuvbvi_hum],
                                                  x_data_err=color_vi_err_hum[mask_class_12_hum * mask_good_colors_nuvbvi_hum],
                                                  y_data_err=color_nuvb_err_hum[mask_class_12_hum * mask_good_colors_nuvbvi_hum],
                                                  x_lim=x_lim_vi, y_lim=y_lim_nuvb, n_bins=n_bins,
                                                  threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

    gauss_dict_ubvi_hum_12 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                                  y_data=color_ub_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                                  x_data_err=color_vi_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                                  y_data_err=color_ub_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                                  x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins,
                                                  threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

    gauss_dict_bvvi_hum_12 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_12_hum * mask_good_colors_bvvi_hum],
                                                  y_data=color_bv_hum[mask_class_12_hum * mask_good_colors_bvvi_hum],
                                                  x_data_err=color_vi_err_hum[mask_class_12_hum * mask_good_colors_bvvi_hum],
                                                  y_data_err=color_bv_err_hum[mask_class_12_hum * mask_good_colors_bvvi_hum],
                                                  x_lim=x_lim_vi, y_lim=y_lim_bv, n_bins=n_bins,
                                                  threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

    mask_ne10 = (ne_mod_sol_neb == 10)
    mask_ne100 = (ne_mod_sol_neb == 100)
    mask_ne1000 = (ne_mod_sol_neb == 1000)

    mask_logu2 = (logu_mod_sol_neb == -2)
    mask_logu3 = (logu_mod_sol_neb == -3)
    mask_logu4 = (logu_mod_sol_neb == -4)

    f_esc_0 = f_esc_mod_sol_neb == 0
    f_esc_25 = f_esc_mod_sol_neb == 0.25
    f_esc_50 = f_esc_mod_sol_neb == 0.50
    f_esc_75 = f_esc_mod_sol_neb == 0.75



    fig, ax = plt.subplots(nrows=3, ncols=4, sharey='row', sharex='col', figsize=(50, 40))
    fontsize = 38

    vmax_nuvbvi = np.nanmax(gauss_dict_nuvbvi_hum_12['gauss_map'])
    ax[0, 0].imshow(gauss_dict_nuvbvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_nuvb[1], y_lim_nuvb[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_nuvbvi/10, vmax=vmax_nuvbvi/1.1)
    ax[0, 1].imshow(gauss_dict_nuvbvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_nuvb[1], y_lim_nuvb[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_nuvbvi/10, vmax=vmax_nuvbvi/1.1)
    ax[0, 2].imshow(gauss_dict_nuvbvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_nuvb[1], y_lim_nuvb[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_nuvbvi/10, vmax=vmax_nuvbvi/1.1)
    ax[0, 3].imshow(gauss_dict_nuvbvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_nuvb[1], y_lim_nuvb[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_nuvbvi/10, vmax=vmax_nuvbvi/1.1)
    vmax_ubvi = np.nanmax(gauss_dict_ubvi_hum_12['gauss_map'])
    ax[1, 0].imshow(gauss_dict_ubvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_ubvi/10, vmax=vmax_ubvi/1.1)
    ax[1, 1].imshow(gauss_dict_ubvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_ubvi/10, vmax=vmax_ubvi/1.1)
    ax[1, 2].imshow(gauss_dict_ubvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_ubvi/10, vmax=vmax_ubvi/1.1)
    ax[1, 3].imshow(gauss_dict_ubvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_ubvi/10, vmax=vmax_ubvi/1.1)
    vmax_bvvi = np.nanmax(gauss_dict_bvvi_hum_12['gauss_map'])
    ax[2, 0].imshow(gauss_dict_bvvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_bvvi/10, vmax=vmax_bvvi/1.1)
    ax[2, 1].imshow(gauss_dict_bvvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_bvvi/10, vmax=vmax_bvvi/1.1)
    ax[2, 2].imshow(gauss_dict_bvvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_bvvi/10, vmax=vmax_bvvi/1.1)
    ax[2, 3].imshow(gauss_dict_bvvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_bv[1], y_lim_bv[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_bvvi/10, vmax=vmax_bvvi/1.1)

    ax[0, 0].set_xlim(x_lim_vi)
    ax[1, 0].set_xlim(x_lim_vi)
    ax[2, 0].set_xlim(x_lim_vi)

    ax[0, 0].set_ylim(y_lim_nuvb)
    ax[1, 0].set_ylim(y_lim_ub)
    ax[2, 0].set_ylim(y_lim_bv)

    # ax[1, 0].set_xlim(-0.7, 1.1)
    # ax[0, 0].set_ylim(-0.2, -2.7)
    # ax[1, 0].set_ylim(-0.4, -2.3)
    # ax[2, 0].set_ylim(0.5, -0.3)


    ax[0, 0].plot(model_vi_sol, model_nuvb_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[0, 1].plot(model_vi_sol, model_nuvb_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[0, 2].plot(model_vi_sol, model_nuvb_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[0, 3].plot(model_vi_sol, model_nuvb_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    # ax[0, 0].plot(model_vi_cb19_sol, model_nuvb_cb19_sol, color='blue', linewidth=3, linestyle='-', label=r'CB19, Z$_{\odot}$')

    ax[1, 0].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[1, 1].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[1, 2].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[1, 3].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    # ax[1, 0].plot(model_vi_cb19_sol, model_ub_cb19_sol, color='blue', linewidth=3, linestyle='-', label=r'CB19, Z$_{\odot}$')

    ax[2, 0].plot(model_vi_sol, model_bv_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[2, 1].plot(model_vi_sol, model_bv_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[2, 2].plot(model_vi_sol, model_bv_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax[2, 3].plot(model_vi_sol, model_bv_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')
    # ax[2, 0].plot(model_vi_cb19_sol, model_bv_cb19_sol, color='blue', linewidth=3, linestyle='-', label=r'CB19, Z$_{\odot}$')


    ax[0, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne10*mask_logu2], model_nuvb_sol_neb[f_esc_0*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
    ax[0, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne10*mask_logu3], model_nuvb_sol_neb[f_esc_0*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
    ax[0, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne10*mask_logu4], model_nuvb_sol_neb[f_esc_0*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

    ax[0, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne100*mask_logu2], model_nuvb_sol_neb[f_esc_0*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
    ax[0, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne100*mask_logu3], model_nuvb_sol_neb[f_esc_0*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
    ax[0, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne100*mask_logu4], model_nuvb_sol_neb[f_esc_0*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

    ax[0, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne1000*mask_logu2], model_nuvb_sol_neb[f_esc_0*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
    ax[0, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne1000*mask_logu3], model_nuvb_sol_neb[f_esc_0*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
    ax[0, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne1000*mask_logu4], model_nuvb_sol_neb[f_esc_0*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')


    ax[0, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne10*mask_logu2], model_nuvb_sol_neb[f_esc_25*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
    ax[0, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne10*mask_logu3], model_nuvb_sol_neb[f_esc_25*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
    ax[0, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne10*mask_logu4], model_nuvb_sol_neb[f_esc_25*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

    ax[0, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne100*mask_logu2], model_nuvb_sol_neb[f_esc_25*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
    ax[0, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne100*mask_logu3], model_nuvb_sol_neb[f_esc_25*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
    ax[0, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne100*mask_logu4], model_nuvb_sol_neb[f_esc_25*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

    ax[0, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne1000*mask_logu2], model_nuvb_sol_neb[f_esc_25*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
    ax[0, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne1000*mask_logu3], model_nuvb_sol_neb[f_esc_25*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
    ax[0, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne1000*mask_logu4], model_nuvb_sol_neb[f_esc_25*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')


    ax[0, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne10*mask_logu2], model_nuvb_sol_neb[f_esc_50*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
    ax[0, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne10*mask_logu3], model_nuvb_sol_neb[f_esc_50*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
    ax[0, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne10*mask_logu4], model_nuvb_sol_neb[f_esc_50*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

    ax[0, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne100*mask_logu2], model_nuvb_sol_neb[f_esc_50*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
    ax[0, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne100*mask_logu3], model_nuvb_sol_neb[f_esc_50*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
    ax[0, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne100*mask_logu4], model_nuvb_sol_neb[f_esc_50*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

    ax[0, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne1000*mask_logu2], model_nuvb_sol_neb[f_esc_50*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
    ax[0, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne1000*mask_logu3], model_nuvb_sol_neb[f_esc_50*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
    ax[0, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne1000*mask_logu4], model_nuvb_sol_neb[f_esc_50*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')


    ax[0, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne10*mask_logu2], model_nuvb_sol_neb[f_esc_75*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
    ax[0, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne10*mask_logu3], model_nuvb_sol_neb[f_esc_75*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
    ax[0, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne10*mask_logu4], model_nuvb_sol_neb[f_esc_75*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

    ax[0, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne100*mask_logu2], model_nuvb_sol_neb[f_esc_75*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
    ax[0, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne100*mask_logu3], model_nuvb_sol_neb[f_esc_75*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
    ax[0, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne100*mask_logu4], model_nuvb_sol_neb[f_esc_75*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

    ax[0, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne1000*mask_logu2], model_nuvb_sol_neb[f_esc_75*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
    ax[0, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne1000*mask_logu3], model_nuvb_sol_neb[f_esc_75*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
    ax[0, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne1000*mask_logu4], model_nuvb_sol_neb[f_esc_75*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')







    ax[1, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne10*mask_logu2], model_ub_sol_neb[f_esc_0*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
    ax[1, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne10*mask_logu3], model_ub_sol_neb[f_esc_0*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
    ax[1, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne10*mask_logu4], model_ub_sol_neb[f_esc_0*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

    ax[1, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne100*mask_logu2], model_ub_sol_neb[f_esc_0*mask_ne100*mask_logu2], marker='o', color='r', s=400)
    ax[1, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne100*mask_logu3], model_ub_sol_neb[f_esc_0*mask_ne100*mask_logu3], marker='o', color='g', s=400)
    ax[1, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne100*mask_logu4], model_ub_sol_neb[f_esc_0*mask_ne100*mask_logu4], marker='o', color='b', s=400)

    ax[1, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne1000*mask_logu2], model_ub_sol_neb[f_esc_0*mask_ne1000*mask_logu2], marker='X', color='r', s=400)
    ax[1, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne1000*mask_logu3], model_ub_sol_neb[f_esc_0*mask_ne1000*mask_logu3], marker='X', color='g', s=400)
    ax[1, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne1000*mask_logu4], model_ub_sol_neb[f_esc_0*mask_ne1000*mask_logu4], marker='X', color='b', s=400)


    ax[1, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne10*mask_logu2], model_ub_sol_neb[f_esc_25*mask_ne10*mask_logu2], marker='P', color='r', s=400)
    ax[1, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne10*mask_logu3], model_ub_sol_neb[f_esc_25*mask_ne10*mask_logu3], marker='P', color='g', s=400)
    ax[1, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne10*mask_logu4], model_ub_sol_neb[f_esc_25*mask_ne10*mask_logu4], marker='P', color='b', s=400)

    ax[1, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne100*mask_logu2], model_ub_sol_neb[f_esc_25*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
    ax[1, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne100*mask_logu3], model_ub_sol_neb[f_esc_25*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
    ax[1, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne100*mask_logu4], model_ub_sol_neb[f_esc_25*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

    ax[1, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne1000*mask_logu2], model_ub_sol_neb[f_esc_25*mask_ne1000*mask_logu2], marker='X', color='r', s=400)
    ax[1, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne1000*mask_logu3], model_ub_sol_neb[f_esc_25*mask_ne1000*mask_logu3], marker='X', color='g', s=400)
    ax[1, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne1000*mask_logu4], model_ub_sol_neb[f_esc_25*mask_ne1000*mask_logu4], marker='X', color='b', s=400)


    ax[1, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne10*mask_logu2], model_ub_sol_neb[f_esc_50*mask_ne10*mask_logu2], marker='P', color='r', s=400)
    ax[1, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne10*mask_logu3], model_ub_sol_neb[f_esc_50*mask_ne10*mask_logu3], marker='P', color='g', s=400)
    ax[1, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne10*mask_logu4], model_ub_sol_neb[f_esc_50*mask_ne10*mask_logu4], marker='P', color='b', s=400)

    ax[1, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne100*mask_logu2], model_ub_sol_neb[f_esc_50*mask_ne100*mask_logu2], marker='o', color='r', s=400)
    ax[1, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne100*mask_logu3], model_ub_sol_neb[f_esc_50*mask_ne100*mask_logu3], marker='o', color='g', s=400)
    ax[1, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne100*mask_logu4], model_ub_sol_neb[f_esc_50*mask_ne100*mask_logu4], marker='o', color='b', s=400)

    ax[1, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne1000*mask_logu2], model_ub_sol_neb[f_esc_50*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
    ax[1, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne1000*mask_logu3], model_ub_sol_neb[f_esc_50*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
    ax[1, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne1000*mask_logu4], model_ub_sol_neb[f_esc_50*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')


    ax[1, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne10*mask_logu2], model_ub_sol_neb[f_esc_75*mask_ne10*mask_logu2], marker='P', color='r', s=400)
    ax[1, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne10*mask_logu3], model_ub_sol_neb[f_esc_75*mask_ne10*mask_logu3], marker='P', color='g', s=400)
    ax[1, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne10*mask_logu4], model_ub_sol_neb[f_esc_75*mask_ne10*mask_logu4], marker='P', color='b', s=400)

    ax[1, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne100*mask_logu2], model_ub_sol_neb[f_esc_75*mask_ne100*mask_logu2], marker='o', color='r', s=400)
    ax[1, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne100*mask_logu3], model_ub_sol_neb[f_esc_75*mask_ne100*mask_logu3], marker='o', color='g', s=400)
    ax[1, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne100*mask_logu4], model_ub_sol_neb[f_esc_75*mask_ne100*mask_logu4], marker='o', color='b', s=400)

    ax[1, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne1000*mask_logu2], model_ub_sol_neb[f_esc_75*mask_ne1000*mask_logu2], marker='X', color='r', s=400)
    ax[1, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne1000*mask_logu3], model_ub_sol_neb[f_esc_75*mask_ne1000*mask_logu3], marker='X', color='g', s=400)
    ax[1, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne1000*mask_logu4], model_ub_sol_neb[f_esc_75*mask_ne1000*mask_logu4], marker='X', color='b', s=400)

    ax[1, 0].legend(frameon=True, loc=1, fontsize=fontsize)
    ax[1, 1].legend(frameon=True, loc=1, fontsize=fontsize)
    ax[1, 2].legend(frameon=True, loc=1, fontsize=fontsize)






    ax[2, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne10*mask_logu2], model_bv_sol_neb[f_esc_0*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
    ax[2, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne10*mask_logu3], model_bv_sol_neb[f_esc_0*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
    ax[2, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne10*mask_logu4], model_bv_sol_neb[f_esc_0*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

    ax[2, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne100*mask_logu2], model_bv_sol_neb[f_esc_0*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
    ax[2, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne100*mask_logu3], model_bv_sol_neb[f_esc_0*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
    ax[2, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne100*mask_logu4], model_bv_sol_neb[f_esc_0*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

    ax[2, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne1000*mask_logu2], model_bv_sol_neb[f_esc_0*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
    ax[2, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne1000*mask_logu3], model_bv_sol_neb[f_esc_0*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
    ax[2, 0].scatter(model_vi_sol_neb[f_esc_0*mask_ne1000*mask_logu4], model_bv_sol_neb[f_esc_0*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')


    ax[2, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne10*mask_logu2], model_bv_sol_neb[f_esc_25*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
    ax[2, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne10*mask_logu3], model_bv_sol_neb[f_esc_25*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
    ax[2, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne10*mask_logu4], model_bv_sol_neb[f_esc_25*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

    ax[2, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne100*mask_logu2], model_bv_sol_neb[f_esc_25*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
    ax[2, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne100*mask_logu3], model_bv_sol_neb[f_esc_25*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
    ax[2, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne100*mask_logu4], model_bv_sol_neb[f_esc_25*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

    ax[2, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne1000*mask_logu2], model_bv_sol_neb[f_esc_25*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
    ax[2, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne1000*mask_logu3], model_bv_sol_neb[f_esc_25*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
    ax[2, 1].scatter(model_vi_sol_neb[f_esc_25*mask_ne1000*mask_logu4], model_bv_sol_neb[f_esc_25*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')


    ax[2, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne10*mask_logu2], model_bv_sol_neb[f_esc_50*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
    ax[2, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne10*mask_logu3], model_bv_sol_neb[f_esc_50*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
    ax[2, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne10*mask_logu4], model_bv_sol_neb[f_esc_50*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

    ax[2, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne100*mask_logu2], model_bv_sol_neb[f_esc_50*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
    ax[2, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne100*mask_logu3], model_bv_sol_neb[f_esc_50*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
    ax[2, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne100*mask_logu4], model_bv_sol_neb[f_esc_50*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

    ax[2, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne1000*mask_logu2], model_bv_sol_neb[f_esc_50*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
    ax[2, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne1000*mask_logu3], model_bv_sol_neb[f_esc_50*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
    ax[2, 2].scatter(model_vi_sol_neb[f_esc_50*mask_ne1000*mask_logu4], model_bv_sol_neb[f_esc_50*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')


    ax[2, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne10*mask_logu2], model_bv_sol_neb[f_esc_75*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
    ax[2, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne10*mask_logu3], model_bv_sol_neb[f_esc_75*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
    ax[2, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne10*mask_logu4], model_bv_sol_neb[f_esc_75*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

    ax[2, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne100*mask_logu2], model_bv_sol_neb[f_esc_75*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
    ax[2, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne100*mask_logu3], model_bv_sol_neb[f_esc_75*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
    ax[2, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne100*mask_logu4], model_bv_sol_neb[f_esc_75*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

    ax[2, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne1000*mask_logu2], model_bv_sol_neb[f_esc_75*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
    ax[2, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne1000*mask_logu3], model_bv_sol_neb[f_esc_75*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
    ax[2, 3].scatter(model_vi_sol_neb[f_esc_75*mask_ne1000*mask_logu4], model_bv_sol_neb[f_esc_75*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')



    vi_int = -0.2
    nuvb_int = -1.25
    ub_int = -0.9
    bv_int = 0.1
    av_value = 1

    helper_func.plot_reddening_vect(ax=ax[0, 0], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                           x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                           linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.0, y_text_offset=-0.05)
    helper_func.plot_reddening_vect(ax=ax[0, 1], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                           x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax[0, 2], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                           x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax[0, 3], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                           x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)

    helper_func.plot_reddening_vect(ax=ax[1, 0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax[1, 1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax[1, 2], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax[1, 3], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)

    helper_func.plot_reddening_vect(ax=ax[2, 0], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                           x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax[2, 1], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                           x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax[2, 2], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                           x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax[2, 3], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                           x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)





    ax[0, 0].set_title(r'f$_{\rm esc}$ = 0.0, E(B-V) = %.1f' % ebv, fontsize=fontsize)
    ax[0, 1].set_title(r'f$_{\rm esc}$ = 0.25, E(B-V) = %.1f' % ebv, fontsize=fontsize)
    ax[0, 2].set_title(r'f$_{\rm esc}$ = 0.50, E(B-V) = %.1f' % ebv, fontsize=fontsize)
    ax[0, 3].set_title(r'f$_{\rm esc}$ = 0.75, E(B-V) = %.1f' % ebv, fontsize=fontsize)




    ax[0, 0].set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
    ax[1, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
    ax[2, 0].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)

    ax[2, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
    ax[2, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
    ax[2, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
    ax[2, 3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


    ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[0, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[1, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[2, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax[2, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


    fig.subplots_adjust(left=0.06, bottom=0.04, right=0.995, top=0.96, wspace=0.01, hspace=0.01)


    plt.savefig('plot_output/ccd_neb_model_ebv_%.1f.png' % ebv)


exit()




figure = plt.figure(figsize=(50, 22))

ax_cc_reg_hum = figure.add_axes([0.12, 0.06, 0.87, 0.93])

vmax = np.nanmax(gauss_map_ubvi_hum_12)
ax_cc_reg_hum.imshow(no_seg_map_ubvi_hum_12, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax/10, vmax=vmax/1.1)
ax_cc_reg_hum.imshow(young_map_ubvi_hum_12, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Blues', vmin=0+vmax/10, vmax=vmax/1.1)
ax_cc_reg_hum.imshow(cascade_map_ubvi_hum_12, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Greens', vmin=0+vmax/10, vmax=vmax/1.1)
ax_cc_reg_hum.imshow(gc_map_ubvi_hum_12, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Reds', vmin=0+vmax/10, vmax=vmax/1.2)


# ax_cc_reg_hum.scatter([], [], color='white', label=r'N = %i' % (sum(mask_class_12_hum)))



# vi_int = model_vi_cb19_sol[age_mod_cb19_sol == 1] # 1.2
# ub_int = model_ub_cb19_sol[age_mod_cb19_sol == 1] # -1.6


# ax_cc_reg_hum.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Class 1|2 (Human)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

# ax_cc_reg_hum.set_title('The PHANGS-HST Bright Star Cluster Sample', fontsize=fontsize)

# ax_cc_reg_hum.set_xlim(x_lim_vi)
# ax_cc_reg_hum.set_ylim(y_lim_ub)
ax_cc_reg_hum.set_xlim(-0.7, 1.1)
ax_cc_reg_hum.set_ylim(-0.4, -2.3)

ax_cc_reg_hum.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=6, linestyle='-', label=r'BC03, Z$_{\odot}$')
ax_cc_reg_hum.plot(model_vi_cb19_sol, model_ub_cb19_sol, color='blue', linewidth=6, linestyle='-', label=r'CB19, Z$_{\odot}$')

mask_ne10 = (ne_mod_sol_neb == 10)
mask_ne100 = (ne_mod_sol_neb == 100)
mask_ne1000 = (ne_mod_sol_neb == 1000)

mask_logu2 = (logu_mod_sol_neb == -2)
mask_logu3 = (logu_mod_sol_neb == -3)
mask_logu4 = (logu_mod_sol_neb == -4)

f_esc_0 = f_esc_mod_sol_neb == 0
f_esc_25 = f_esc_mod_sol_neb == 0.25
f_esc_50 = f_esc_mod_sol_neb == 0.50
f_esc_75 = f_esc_mod_sol_neb == 0.75
f_esc_100 = f_esc_mod_sol_neb == 1.00

model_str = 100
f_esc_mask = globals()['f_esc_%s' % model_str]


pe = [patheffects.withStroke(linewidth=3, foreground="w")]

ax_cc_reg_hum.text(0.02, 0.95, r'f$_{\rm esc}$ = %.2f' % (model_str/100), horizontalalignment='left',
                   verticalalignment='center', fontsize=fontsize,
              transform=ax_cc_reg_hum.transAxes, path_effects=pe)

ax_cc_reg_hum.scatter(model_vi_sol_neb[f_esc_mask*mask_ne10*mask_logu2], model_ub_sol_neb[f_esc_mask*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
ax_cc_reg_hum.scatter(model_vi_sol_neb[f_esc_mask*mask_ne10*mask_logu3], model_ub_sol_neb[f_esc_mask*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-3')
ax_cc_reg_hum.scatter(model_vi_sol_neb[f_esc_mask*mask_ne10*mask_logu4], model_ub_sol_neb[f_esc_mask*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-4')

ax_cc_reg_hum.scatter(model_vi_sol_neb[f_esc_mask*mask_ne100*mask_logu2], model_ub_sol_neb[f_esc_mask*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
ax_cc_reg_hum.scatter(model_vi_sol_neb[f_esc_mask*mask_ne100*mask_logu3], model_ub_sol_neb[f_esc_mask*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-3')
ax_cc_reg_hum.scatter(model_vi_sol_neb[f_esc_mask*mask_ne100*mask_logu4], model_ub_sol_neb[f_esc_mask*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-4')

ax_cc_reg_hum.scatter(model_vi_sol_neb[f_esc_mask*mask_ne1000*mask_logu2], model_ub_sol_neb[f_esc_mask*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
ax_cc_reg_hum.scatter(model_vi_sol_neb[f_esc_mask*mask_ne1000*mask_logu3], model_ub_sol_neb[f_esc_mask*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
ax_cc_reg_hum.scatter(model_vi_sol_neb[f_esc_mask*mask_ne1000*mask_logu4], model_ub_sol_neb[f_esc_mask*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')



vi_int = -0.4
ub_int = -1.0
av_value = 1

helper_func.plot_reddening_vect(ax=ax_cc_reg_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.00, y_text_offset=-0.05,
                                linewidth=6, line_color='k', text=True, fontsize=fontsize)

ax_cc_reg_hum.legend(frameon=True, loc=1, fontsize=fontsize)
ax_cc_reg_hum.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_cc_reg_hum.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


ax_cc_reg_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

# plt.tight_layout()
plt.savefig('plot_output/cc_nebular_models_f_esc_%i.png' % model_str)
figure.clf()
ax_cc_reg_hum.cla()
# plt.close()



fig, ax_ew_age = plt.subplots(figsize=(15, 15))
fontsize = 30
ax_ew_age.scatter(age_mod_sol_neb[mask_ne10*mask_logu2], ew_h_alpha[mask_ne10*mask_logu2],
                  marker='P', color='r', s=400, label='ne=10, logU=-2')
ax_ew_age.scatter(age_mod_sol_neb[mask_ne10*mask_logu3], ew_h_alpha[mask_ne10*mask_logu3],
                  marker='P', color='g', s=200, label='ne=10, logU=-3')
ax_ew_age.scatter(age_mod_sol_neb[mask_ne10*mask_logu4], ew_h_alpha[mask_ne10*mask_logu4],
                  marker='P', color='b', s=200, label='ne=10, logU=-4')

ax_ew_age.scatter(age_mod_sol_neb[mask_ne100*mask_logu2], ew_h_alpha[mask_ne100*mask_logu2],
                  marker='o', color='r', s=200, label='ne=100, logU=-2')
ax_ew_age.scatter(age_mod_sol_neb[mask_ne100*mask_logu3], ew_h_alpha[mask_ne100*mask_logu3],
                  marker='o', color='g', s=200, label='ne=100, logU=-3')
ax_ew_age.scatter(age_mod_sol_neb[mask_ne100*mask_logu4], ew_h_alpha[mask_ne100*mask_logu4],
                  marker='o', color='b', s=200, label='ne=100, logU=-4')

ax_ew_age.scatter(age_mod_sol_neb[mask_ne1000*mask_logu2], ew_h_alpha[mask_ne1000*mask_logu2],
                  marker='X', color='r', s=200, label='ne=1000, logU=-2')
ax_ew_age.scatter(age_mod_sol_neb[mask_ne1000*mask_logu3], ew_h_alpha[mask_ne1000*mask_logu3],
                  marker='X', color='g', s=200, label='ne=1000, logU=-3')
ax_ew_age.scatter(age_mod_sol_neb[mask_ne1000*mask_logu4], ew_h_alpha[mask_ne1000*mask_logu4],
                  marker='X', color='b', s=200, label='ne=1000, logU=-4')

ax_ew_age.set_xlim(0, 7.9)
ax_ew_age.legend(frameon=True, loc=1, fontsize=fontsize)
ax_ew_age.set_ylabel(r'EW(H$\alpha$) [$\AA$]', fontsize=fontsize)
ax_ew_age.set_xlabel('Age [Myr]', fontsize=fontsize)

ax_ew_age.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(left=0.11, bottom=0.07, right=0.995, top=0.95, wspace=0.01, hspace=0.01)

plt.savefig('plot_output/we_age.png')




