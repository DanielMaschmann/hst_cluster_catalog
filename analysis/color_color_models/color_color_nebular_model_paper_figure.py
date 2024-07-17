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

x_lim_vi = (-0.7, 1.2)

y_lim_nuvb = (3.2, -2.9)
y_lim_ub = (-0.3, -2.3)
y_lim_bv = (1.9, -0.7)

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

model_nuvb_sol_neb = mag_nuv_sol_neb - mag_b_sol_neb
model_ub_sol_neb = mag_u_sol_neb - mag_b_sol_neb
model_bv_sol_neb = mag_b_sol_neb - mag_v_sol_neb
model_vi_sol_neb = mag_v_sol_neb - mag_i_sol_neb



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

f_esc = 0.00

f_esc_mask = f_esc_mod_sol_neb == f_esc

mask_age = age_mod_sol_neb < 7


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(17, 17))
fontsize = 30

vmax_ubvi = np.nanmax(gauss_dict_ubvi_hum_12['gauss_map'])
ax.imshow(gauss_dict_ubvi_hum_12['gauss_map'], origin='lower',
                    extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax_ubvi/10, vmax=vmax_ubvi/1.1)

ax.set_xlim(x_lim_vi)
ax.set_ylim(y_lim_ub)

ax.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=3, linestyle='-', label=r'BC03, Z$_{\odot}$')

# ax.plot([], [], color='white', label=r'f$_{\rm esc}$ = %.2f' % f_esc)

ax.scatter(model_vi_sol_neb[mask_age*f_esc_mask*mask_ne10*mask_logu2], model_ub_sol_neb[mask_age*f_esc_mask*mask_ne10*mask_logu2], marker='P', color='r', s=400, label='ne=10, logU=-2')
ax.scatter(model_vi_sol_neb[mask_age*f_esc_mask*mask_ne10*mask_logu3], model_ub_sol_neb[mask_age*f_esc_mask*mask_ne10*mask_logu3], marker='P', color='g', s=400, label='ne=10, logU=-2')
ax.scatter(model_vi_sol_neb[mask_age*f_esc_mask*mask_ne10*mask_logu4], model_ub_sol_neb[mask_age*f_esc_mask*mask_ne10*mask_logu4], marker='P', color='b', s=400, label='ne=10, logU=-2')

ax.scatter(model_vi_sol_neb[mask_age*f_esc_mask*mask_ne100*mask_logu2], model_ub_sol_neb[mask_age*f_esc_mask*mask_ne100*mask_logu2], marker='o', color='r', s=400, label='ne=100, logU=-2')
ax.scatter(model_vi_sol_neb[mask_age*f_esc_mask*mask_ne100*mask_logu3], model_ub_sol_neb[mask_age*f_esc_mask*mask_ne100*mask_logu3], marker='o', color='g', s=400, label='ne=100, logU=-2')
ax.scatter(model_vi_sol_neb[mask_age*f_esc_mask*mask_ne100*mask_logu4], model_ub_sol_neb[mask_age*f_esc_mask*mask_ne100*mask_logu4], marker='o', color='b', s=400, label='ne=100, logU=-2')

ax.scatter(model_vi_sol_neb[mask_age*f_esc_mask*mask_ne1000*mask_logu2], model_ub_sol_neb[mask_age*f_esc_mask*mask_ne1000*mask_logu2], marker='X', color='r', s=400, label='ne=1000, logU=-2')
ax.scatter(model_vi_sol_neb[mask_age*f_esc_mask*mask_ne1000*mask_logu3], model_ub_sol_neb[mask_age*f_esc_mask*mask_ne1000*mask_logu3], marker='X', color='g', s=400, label='ne=1000, logU=-3')
ax.scatter(model_vi_sol_neb[mask_age*f_esc_mask*mask_ne1000*mask_logu4], model_ub_sol_neb[mask_age*f_esc_mask*mask_ne1000*mask_logu4], marker='X', color='b', s=400, label='ne=1000, logU=-4')

ax.legend(frameon=True, loc=1, fontsize=fontsize)

vi_int = -0.2
nuvb_int = -1.25
ub_int = -0.9
bv_int = 0.1
av_value = 1

helper_func.plot_reddening_vect(ax=ax, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize,
                                x_text_offset=0.12, y_text_offset=0.02)


ax.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

# ax.set_title(r'f$_{\rm esc}$ = %.2f' % f_esc, fontsize=fontsize)
ax.text(0.03, 0.97, r'f$_{\rm esc}$ = %.2f' % f_esc,
              horizontalalignment='left', verticalalignment='top', color='k',
              fontsize=fontsize, transform=ax.transAxes)

ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.tight_layout()

plt.savefig('plot_output/ccd_neb_model_f_esc_%i.png' % int(f_esc * 100))
plt.savefig('plot_output/ccd_neb_model_f_esc_%i.pdf' % int(f_esc * 100))

